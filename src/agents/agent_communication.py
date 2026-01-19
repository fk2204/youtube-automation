"""
Agent Communication System - Inter-agent Communication Infrastructure

Provides robust communication infrastructure for agent collaboration including
message passing, event broadcasting, shared state, task queuing, and result caching.

Components:
- MessageBus: Pub/sub message passing between agents
- EventEmitter: Broadcast events to interested agents
- SharedState: Thread-safe shared state management
- TaskQueue: Priority-based task queue
- ResultCache: Cache agent results with TTL

Usage:
    from src.agents.agent_communication import (
        get_message_bus,
        get_event_emitter,
        get_shared_state,
        get_task_queue,
        get_result_cache,
    )

    # Message passing
    bus = get_message_bus()
    bus.subscribe("research_complete", handler)
    bus.publish("research_complete", {"topics": [...]})

    # Events
    emitter = get_event_emitter()
    emitter.on("workflow_started", handler)
    emitter.emit("workflow_started", workflow_id="wf_123")

    # Shared state
    state = get_shared_state()
    state.set("current_workflow", {"id": "wf_123", "status": "running"})
    workflow = state.get("current_workflow")

    # Task queue
    queue = get_task_queue()
    queue.enqueue("research", {"niche": "finance"}, priority=2)
    task = queue.dequeue()

    # Result cache
    cache = get_result_cache()
    cache.set("research_finance", results, ttl=3600)
    cached = cache.get("research_finance")
"""

import asyncio
import hashlib
import json
import pickle
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import PriorityQueue, Empty
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from uuid import uuid4

from loguru import logger


# ============================================================================
# Type Variables
# ============================================================================

T = TypeVar("T")


# ============================================================================
# Data Classes and Enums
# ============================================================================


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class Message:
    """Message for inter-agent communication."""
    topic: str
    payload: Dict[str, Any]
    sender: str = "unknown"
    message_id: str = field(default_factory=lambda: uuid4().hex)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "message_id": self.message_id,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            topic=data["topic"],
            payload=data["payload"],
            sender=data.get("sender", "unknown"),
            message_id=data.get("message_id", uuid4().hex),
            priority=MessagePriority(data.get("priority", 3)),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl_seconds=data.get("ttl_seconds"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Event:
    """Event for broadcasting to agents."""
    event_type: str
    data: Dict[str, Any]
    source: str = "system"
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.now)
    propagate: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "propagate": self.propagate,
        }


@dataclass
class QueuedTask:
    """Task in the priority queue."""
    task_id: str
    task_type: str
    params: Dict[str, Any]
    priority: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "QueuedTask") -> bool:
        # Lower priority number = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # Earlier deadline = higher priority
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        return self.created_at < other.created_at


@dataclass
class CachedResult:
    """Cached result with TTL."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


# ============================================================================
# Message Bus
# ============================================================================


class MessageBus:
    """
    Pub/sub message bus for agent communication.

    Features:
    - Topic-based subscriptions
    - Async and sync message handling
    - Message filtering
    - Dead letter queue
    - Message persistence (optional)
    """

    def __init__(self, persist: bool = False, db_path: Optional[str] = None):
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self._async_subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self._filters: Dict[str, List[Callable[[Message], bool]]] = defaultdict(list)
        self._dead_letters: List[Message] = []
        self._message_history: List[Message] = []
        self._lock = threading.RLock()
        self._persist = persist
        self._db_path = db_path or "data/message_bus.db"

        if persist:
            self._init_db()

        logger.info("MessageBus initialized")

    def _init_db(self):
        """Initialize persistence database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    topic TEXT,
                    payload TEXT,
                    sender TEXT,
                    priority INTEGER,
                    timestamp TEXT,
                    processed INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic ON messages(topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed ON messages(processed)
            """)

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], None],
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """
        Subscribe to a topic.

        Args:
            topic: Topic to subscribe to (supports wildcards: "research.*")
            handler: Callback function for messages
            filter_fn: Optional filter function

        Returns:
            Subscription ID
        """
        with self._lock:
            self._subscriptions[topic].append(handler)
            if filter_fn:
                self._filters[topic].append(filter_fn)

            sub_id = f"sub_{topic}_{uuid4().hex[:8]}"
            logger.debug(f"Subscribed to topic: {topic} ({sub_id})")
            return sub_id

    def subscribe_async(
        self,
        topic: str,
        handler: Callable[[Message], Any],
    ) -> str:
        """Subscribe with an async handler."""
        with self._lock:
            self._async_subscriptions[topic].append(handler)
            sub_id = f"async_sub_{topic}_{uuid4().hex[:8]}"
            logger.debug(f"Async subscribed to topic: {topic} ({sub_id})")
            return sub_id

    def unsubscribe(self, topic: str, handler: Callable) -> bool:
        """Unsubscribe from a topic."""
        with self._lock:
            if topic in self._subscriptions:
                if handler in self._subscriptions[topic]:
                    self._subscriptions[topic].remove(handler)
                    return True
            if topic in self._async_subscriptions:
                if handler in self._async_subscriptions[topic]:
                    self._async_subscriptions[topic].remove(handler)
                    return True
            return False

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender: str = "unknown",
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> Message:
        """
        Publish a message to a topic.

        Args:
            topic: Topic to publish to
            payload: Message payload
            sender: Sender identifier
            priority: Message priority
            correlation_id: For request/response correlation

        Returns:
            Published message
        """
        message = Message(
            topic=topic,
            payload=payload,
            sender=sender,
            priority=priority,
            correlation_id=correlation_id,
        )

        with self._lock:
            self._message_history.append(message)

            # Persist if enabled
            if self._persist:
                self._persist_message(message)

            # Find matching subscriptions
            matching_topics = self._find_matching_topics(topic)
            delivered = False

            for match_topic in matching_topics:
                # Check filters
                filters = self._filters.get(match_topic, [])
                if filters and not all(f(message) for f in filters):
                    continue

                # Sync handlers
                for handler in self._subscriptions.get(match_topic, []):
                    try:
                        handler(message)
                        delivered = True
                    except Exception as e:
                        logger.error(f"Handler error for topic {topic}: {e}")

                # Async handlers
                for handler in self._async_subscriptions.get(match_topic, []):
                    try:
                        asyncio.create_task(handler(message))
                        delivered = True
                    except Exception as e:
                        logger.error(f"Async handler error for topic {topic}: {e}")

            if not delivered:
                self._dead_letters.append(message)
                logger.warning(f"No subscribers for topic: {topic}")

        return message

    async def publish_async(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender: str = "unknown",
        timeout: float = 5.0,
    ) -> List[Any]:
        """
        Publish and wait for async responses.

        Args:
            topic: Topic to publish to
            payload: Message payload
            sender: Sender identifier
            timeout: Maximum wait time

        Returns:
            List of handler responses
        """
        message = Message(
            topic=topic,
            payload=payload,
            sender=sender,
        )

        responses = []
        matching_topics = self._find_matching_topics(topic)

        async def collect_response(handler, msg):
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await asyncio.wait_for(handler(msg), timeout)
                else:
                    return handler(msg)
            except asyncio.TimeoutError:
                logger.warning(f"Handler timeout for topic: {topic}")
                return None
            except Exception as e:
                logger.error(f"Handler error: {e}")
                return None

        tasks = []
        for match_topic in matching_topics:
            for handler in self._async_subscriptions.get(match_topic, []):
                tasks.append(collect_response(handler, message))
            for handler in self._subscriptions.get(match_topic, []):
                tasks.append(collect_response(handler, message))

        if tasks:
            responses = await asyncio.gather(*tasks)

        return [r for r in responses if r is not None]

    def _find_matching_topics(self, topic: str) -> List[str]:
        """Find all subscription topics matching the published topic."""
        matches = []
        with self._lock:
            for sub_topic in list(self._subscriptions.keys()) + list(self._async_subscriptions.keys()):
                if self._topic_matches(sub_topic, topic):
                    if sub_topic not in matches:
                        matches.append(sub_topic)
        return matches

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if topic matches pattern (supports wildcards)."""
        if pattern == topic:
            return True

        # Handle wildcards
        if "*" in pattern:
            pattern_parts = pattern.split(".")
            topic_parts = topic.split(".")

            if len(pattern_parts) > len(topic_parts):
                return False

            for i, p in enumerate(pattern_parts):
                if p == "*":
                    continue
                if p == "**":
                    return True
                if i >= len(topic_parts) or p != topic_parts[i]:
                    return False
            return True

        return False

    def _persist_message(self, message: Message):
        """Persist message to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO messages (message_id, topic, payload, sender, priority, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.topic,
                        json.dumps(message.payload),
                        message.sender,
                        message.priority.value,
                        message.timestamp.isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

    def get_dead_letters(self) -> List[Message]:
        """Get all dead letter messages."""
        with self._lock:
            return list(self._dead_letters)

    def clear_dead_letters(self) -> int:
        """Clear dead letter queue."""
        with self._lock:
            count = len(self._dead_letters)
            self._dead_letters.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        with self._lock:
            return {
                "total_subscriptions": sum(
                    len(h) for h in self._subscriptions.values()
                ) + sum(len(h) for h in self._async_subscriptions.values()),
                "topics": list(set(self._subscriptions.keys()) | set(self._async_subscriptions.keys())),
                "messages_processed": len(self._message_history),
                "dead_letters": len(self._dead_letters),
            }


# ============================================================================
# Event Emitter
# ============================================================================


class EventEmitter:
    """
    Event emitter for broadcasting events to interested agents.

    Features:
    - Event type registration
    - One-time listeners
    - Event history
    - Async event handling
    """

    def __init__(self, max_history: int = 1000):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._once_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        logger.info("EventEmitter initialized")

    def on(self, event_type: str, handler: Callable[[Event], Any]) -> str:
        """
        Register a listener for an event type.

        Args:
            event_type: Type of event to listen for
            handler: Callback function

        Returns:
            Listener ID
        """
        with self._lock:
            self._listeners[event_type].append(handler)
            listener_id = f"listener_{event_type}_{uuid4().hex[:8]}"
            logger.debug(f"Registered listener for: {event_type}")
            return listener_id

    def once(self, event_type: str, handler: Callable[[Event], Any]) -> str:
        """Register a one-time listener."""
        with self._lock:
            self._once_listeners[event_type].append(handler)
            listener_id = f"once_{event_type}_{uuid4().hex[:8]}"
            logger.debug(f"Registered one-time listener for: {event_type}")
            return listener_id

    def off(self, event_type: str, handler: Optional[Callable] = None) -> int:
        """
        Remove listeners for an event type.

        Args:
            event_type: Event type
            handler: Specific handler to remove (None = remove all)

        Returns:
            Number of listeners removed
        """
        with self._lock:
            removed = 0
            if handler is None:
                removed = len(self._listeners.get(event_type, []))
                self._listeners[event_type] = []
                removed += len(self._once_listeners.get(event_type, []))
                self._once_listeners[event_type] = []
            else:
                if handler in self._listeners.get(event_type, []):
                    self._listeners[event_type].remove(handler)
                    removed += 1
                if handler in self._once_listeners.get(event_type, []):
                    self._once_listeners[event_type].remove(handler)
                    removed += 1
            return removed

    def emit(
        self,
        event_type: str,
        source: str = "system",
        propagate: bool = True,
        **data,
    ) -> Event:
        """
        Emit an event.

        Args:
            event_type: Type of event
            source: Event source
            propagate: Whether to propagate to parent types
            **data: Event data

        Returns:
            Emitted event
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            propagate=propagate,
        )

        with self._lock:
            # Add to history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Call regular listeners
            for handler in self._listeners.get(event_type, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")

            # Call once listeners
            once_handlers = self._once_listeners.get(event_type, [])
            self._once_listeners[event_type] = []
            for handler in once_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Once handler error for {event_type}: {e}")

            # Propagate to wildcard listeners
            if propagate:
                for listener_type in list(self._listeners.keys()):
                    if listener_type.endswith("*"):
                        prefix = listener_type[:-1]
                        if event_type.startswith(prefix):
                            for handler in self._listeners[listener_type]:
                                try:
                                    handler(event)
                                except Exception as e:
                                    logger.error(f"Wildcard handler error: {e}")

        return event

    async def emit_async(
        self,
        event_type: str,
        timeout: float = 5.0,
        **data,
    ) -> List[Any]:
        """Emit event and collect async responses."""
        event = Event(event_type=event_type, data=data)

        async def call_handler(handler):
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await asyncio.wait_for(handler(event), timeout)
                else:
                    return handler(event)
            except Exception as e:
                logger.error(f"Async handler error: {e}")
                return None

        tasks = []
        with self._lock:
            for handler in self._listeners.get(event_type, []):
                tasks.append(call_handler(handler))

        if tasks:
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

        return []

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history."""
        with self._lock:
            events = self._history
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        with self._lock:
            return {
                "event_types": list(self._listeners.keys()),
                "total_listeners": sum(len(h) for h in self._listeners.values()),
                "once_listeners": sum(len(h) for h in self._once_listeners.values()),
                "history_size": len(self._history),
            }


# ============================================================================
# Shared State
# ============================================================================


class SharedState:
    """
    Thread-safe shared state management.

    Features:
    - Thread-safe read/write
    - Namespace isolation
    - Change notifications
    - State persistence
    """

    def __init__(self, persist: bool = False, db_path: Optional[str] = None):
        self._state: Dict[str, Any] = {}
        self._namespaces: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._persist = persist
        self._db_path = db_path or "data/shared_state.db"

        if persist:
            self._init_db()
            self._load_state()

        logger.info("SharedState initialized")

    def _init_db(self):
        """Initialize persistence database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    namespace TEXT DEFAULT 'default',
                    updated_at TEXT
                )
            """)

    def _load_state(self):
        """Load state from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("SELECT key, value, namespace FROM state").fetchall()
                for key, value, namespace in rows:
                    try:
                        decoded = pickle.loads(value)
                        if namespace == "default":
                            self._state[key] = decoded
                        else:
                            self._namespaces[namespace][key] = decoded
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        notify: bool = True,
    ) -> None:
        """
        Set a value in shared state.

        Args:
            key: State key
            value: Value to store
            namespace: Namespace for isolation
            notify: Whether to notify watchers
        """
        with self._lock:
            old_value = self.get(key, namespace=namespace)

            if namespace == "default":
                self._state[key] = value
            else:
                self._namespaces[namespace][key] = value

            # Persist if enabled
            if self._persist:
                self._persist_value(key, value, namespace)

            # Notify watchers
            if notify and old_value != value:
                watch_key = f"{namespace}:{key}" if namespace != "default" else key
                for watcher in self._watchers.get(watch_key, []):
                    try:
                        watcher(key, old_value, value)
                    except Exception as e:
                        logger.error(f"Watcher error for {key}: {e}")

    def get(
        self,
        key: str,
        default: Any = None,
        namespace: str = "default",
    ) -> Any:
        """
        Get a value from shared state.

        Args:
            key: State key
            default: Default value if not found
            namespace: Namespace to look in

        Returns:
            Stored value or default
        """
        with self._lock:
            if namespace == "default":
                return self._state.get(key, default)
            else:
                return self._namespaces.get(namespace, {}).get(key, default)

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a value from shared state."""
        with self._lock:
            if namespace == "default":
                if key in self._state:
                    del self._state[key]
                    if self._persist:
                        self._delete_value(key, namespace)
                    return True
            else:
                if key in self._namespaces.get(namespace, {}):
                    del self._namespaces[namespace][key]
                    if self._persist:
                        self._delete_value(key, namespace)
                    return True
            return False

    def update(
        self,
        key: str,
        updates: Dict[str, Any],
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """
        Update a dictionary value.

        Args:
            key: State key
            updates: Dictionary of updates
            namespace: Namespace

        Returns:
            Updated value
        """
        with self._lock:
            current = self.get(key, {}, namespace)
            if isinstance(current, dict):
                current.update(updates)
                self.set(key, current, namespace)
            return current

    def watch(
        self,
        key: str,
        callback: Callable[[str, Any, Any], None],
        namespace: str = "default",
    ) -> str:
        """
        Watch for changes to a key.

        Args:
            key: Key to watch
            callback: Callback(key, old_value, new_value)
            namespace: Namespace

        Returns:
            Watcher ID
        """
        watch_key = f"{namespace}:{key}" if namespace != "default" else key
        with self._lock:
            self._watchers[watch_key].append(callback)
            watcher_id = f"watcher_{watch_key}_{uuid4().hex[:8]}"
            return watcher_id

    def unwatch(self, key: str, callback: Callable, namespace: str = "default") -> bool:
        """Remove a watcher."""
        watch_key = f"{namespace}:{key}" if namespace != "default" else key
        with self._lock:
            if callback in self._watchers.get(watch_key, []):
                self._watchers[watch_key].remove(callback)
                return True
            return False

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all values in a namespace."""
        with self._lock:
            if namespace == "default":
                return dict(self._state)
            return dict(self._namespaces.get(namespace, {}))

    def clear_namespace(self, namespace: str) -> int:
        """Clear all values in a namespace."""
        with self._lock:
            if namespace == "default":
                count = len(self._state)
                self._state.clear()
            else:
                count = len(self._namespaces.get(namespace, {}))
                self._namespaces[namespace].clear()
            return count

    def _persist_value(self, key: str, value: Any, namespace: str):
        """Persist value to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO state (key, value, namespace, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, pickle.dumps(value), namespace, datetime.now().isoformat()),
                )
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    def _delete_value(self, key: str, namespace: str):
        """Delete value from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "DELETE FROM state WHERE key = ? AND namespace = ?",
                    (key, namespace),
                )
        except Exception as e:
            logger.error(f"Failed to delete state: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get state statistics."""
        with self._lock:
            return {
                "default_keys": len(self._state),
                "namespaces": list(self._namespaces.keys()),
                "namespace_counts": {
                    ns: len(vals) for ns, vals in self._namespaces.items()
                },
                "watchers": len(self._watchers),
            }


# ============================================================================
# Task Queue
# ============================================================================


class TaskQueue:
    """
    Priority-based task queue.

    Features:
    - Priority ordering
    - Deadline support
    - Retry handling
    - Persistence
    - Async support
    """

    def __init__(
        self,
        max_size: int = 10000,
        persist: bool = False,
        db_path: Optional[str] = None,
    ):
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_size)
        self._tasks: Dict[str, QueuedTask] = {}
        self._processing: Set[str] = set()
        self._completed: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._persist = persist
        self._db_path = db_path or "data/task_queue.db"

        if persist:
            self._init_db()
            self._load_tasks()

        logger.info("TaskQueue initialized")

    def _init_db(self):
        """Initialize persistence database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT,
                    params TEXT,
                    priority INTEGER,
                    created_at TEXT,
                    deadline TEXT,
                    retries INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)
            """)

    def _load_tasks(self):
        """Load pending tasks from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT task_id, task_type, params, priority, created_at, deadline, retries "
                    "FROM tasks WHERE status = 'pending'"
                ).fetchall()
                for row in rows:
                    task = QueuedTask(
                        task_id=row[0],
                        task_type=row[1],
                        params=json.loads(row[2]),
                        priority=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        deadline=datetime.fromisoformat(row[5]) if row[5] else None,
                        retries=row[6],
                    )
                    self._tasks[task.task_id] = task
                    self._queue.put(task)
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def enqueue(
        self,
        task_type: str,
        params: Dict[str, Any],
        priority: int = 3,
        deadline: Optional[datetime] = None,
        task_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Add a task to the queue.

        Args:
            task_type: Type of task
            params: Task parameters
            priority: Priority (1=highest, 5=lowest)
            deadline: Optional deadline
            task_id: Optional custom task ID
            max_retries: Maximum retry attempts

        Returns:
            Task ID
        """
        task_id = task_id or f"task_{uuid4().hex}"

        task = QueuedTask(
            task_id=task_id,
            task_type=task_type,
            params=params,
            priority=priority,
            deadline=deadline,
            max_retries=max_retries,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._queue.put(task)

            if self._persist:
                self._persist_task(task)

        logger.debug(f"Enqueued task: {task_id} (priority={priority})")
        return task_id

    def dequeue(self, timeout: Optional[float] = None) -> Optional[QueuedTask]:
        """
        Get the next task from the queue.

        Args:
            timeout: Maximum wait time

        Returns:
            Next task or None
        """
        try:
            task = self._queue.get(block=True, timeout=timeout)

            with self._lock:
                self._processing.add(task.task_id)

                if self._persist:
                    self._update_task_status(task.task_id, "processing")

            return task

        except Empty:
            return None

    def complete(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self._processing:
                return False

            self._processing.discard(task_id)
            self._completed[task_id] = result

            if task_id in self._tasks:
                del self._tasks[task_id]

            if self._persist:
                self._update_task_status(task_id, "completed")

            logger.debug(f"Completed task: {task_id}")
            return True

    def fail(self, task_id: str, error: str = "") -> bool:
        """
        Mark a task as failed.

        Will re-queue if retries remaining.
        """
        with self._lock:
            if task_id not in self._processing:
                return False

            self._processing.discard(task_id)
            task = self._tasks.get(task_id)

            if task and task.retries < task.max_retries:
                # Re-queue with increased retry count
                task.retries += 1
                self._queue.put(task)
                logger.warning(f"Task {task_id} failed, retry {task.retries}/{task.max_retries}")

                if self._persist:
                    self._update_task_retries(task_id, task.retries)
                return True
            else:
                # Max retries exceeded
                if task_id in self._tasks:
                    del self._tasks[task_id]

                if self._persist:
                    self._update_task_status(task_id, "failed")

                logger.error(f"Task {task_id} failed permanently: {error}")
                return False

    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()

    def processing_count(self) -> int:
        """Get number of tasks being processed."""
        with self._lock:
            return len(self._processing)

    def peek(self) -> Optional[QueuedTask]:
        """Peek at the next task without removing it."""
        with self._lock:
            for task in sorted(self._tasks.values()):
                if task.task_id not in self._processing:
                    return task
            return None

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task."""
        with self._lock:
            if task_id in self._tasks and task_id not in self._processing:
                del self._tasks[task_id]

                if self._persist:
                    self._update_task_status(task_id, "cancelled")

                return True
            return False

    def clear(self) -> int:
        """Clear all pending tasks."""
        with self._lock:
            count = len(self._tasks)
            self._tasks.clear()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            return count

    def _persist_task(self, task: QueuedTask):
        """Persist task to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks
                    (task_id, task_type, params, priority, created_at, deadline, retries, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        task.task_id,
                        task.task_type,
                        json.dumps(task.params),
                        task.priority,
                        task.created_at.isoformat(),
                        task.deadline.isoformat() if task.deadline else None,
                        task.retries,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to persist task: {e}")

    def _update_task_status(self, task_id: str, status: str):
        """Update task status in database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (status, task_id),
                )
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

    def _update_task_retries(self, task_id: str, retries: int):
        """Update task retries in database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE tasks SET retries = ? WHERE task_id = ?",
                    (retries, task_id),
                )
        except Exception as e:
            logger.error(f"Failed to update task retries: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                "pending": len(self._tasks),
                "processing": len(self._processing),
                "completed": len(self._completed),
                "queue_size": self._queue.qsize(),
            }


# ============================================================================
# Result Cache
# ============================================================================


class ResultCache:
    """
    Cache agent results with TTL.

    Features:
    - Time-based expiration
    - LRU eviction
    - Key namespacing
    - Hit/miss statistics
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        persist: bool = False,
        db_path: Optional[str] = None,
    ):
        self._cache: Dict[str, CachedResult] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
        self._persist = persist
        self._db_path = db_path or "data/result_cache.db"

        if persist:
            self._init_db()

        logger.info(f"ResultCache initialized (max_size={max_size}, ttl={default_ttl}s)")

    def _init_db(self):
        """Initialize persistence database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = default)
            metadata: Optional metadata
        """
        ttl = ttl if ttl is not None else self._default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

        cached = CachedResult(
            key=key,
            value=value,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            self._cache[key] = cached

            if self._persist:
                self._persist_value(key, cached)

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        with self._lock:
            cached = self._cache.get(key)

            if cached is None:
                self._misses += 1
                return default

            if cached.is_expired:
                del self._cache[key]
                self._misses += 1
                return default

            cached.hit_count += 1
            self._hits += 1
            return cached.value

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """Async version of get_or_set."""
        value = self.get(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        self.set(key, value, ttl)
        return value

    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self._persist:
                    self._delete_value(key)
                return True
            return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached values.

        Args:
            pattern: Optional key pattern (prefix match)

        Returns:
            Number of items cleared
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
            else:
                keys_to_delete = [
                    k for k in self._cache if k.startswith(pattern)
                ]
                count = len(keys_to_delete)
                for key in keys_to_delete:
                    del self._cache[key]
            return count

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return False
            if cached.is_expired:
                del self._cache[key]
                return False
            return True

    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key in seconds."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is None or cached.expires_at is None:
                return -1
            remaining = (cached.expires_at - datetime.now()).total_seconds()
            return max(0, int(remaining))

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._cache:
            return

        # Find item with lowest hit count
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].hit_count,
        )
        del self._cache[lru_key]

    def _persist_value(self, key: str, cached: CachedResult):
        """Persist value to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        key,
                        pickle.dumps(cached.value),
                        cached.created_at.isoformat(),
                        cached.expires_at.isoformat() if cached.expires_at else None,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to persist cache: {e}")

    def _delete_value(self, key: str):
        """Delete value from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")

    def cleanup_expired(self) -> int:
        """Remove all expired items."""
        with self._lock:
            expired = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired:
                del self._cache[key]
            return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


# ============================================================================
# Singleton Instances
# ============================================================================


_message_bus: Optional[MessageBus] = None
_event_emitter: Optional[EventEmitter] = None
_shared_state: Optional[SharedState] = None
_task_queue: Optional[TaskQueue] = None
_result_cache: Optional[ResultCache] = None


def get_message_bus(persist: bool = False) -> MessageBus:
    """Get or create global MessageBus instance."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus(persist=persist)
    return _message_bus


def get_event_emitter() -> EventEmitter:
    """Get or create global EventEmitter instance."""
    global _event_emitter
    if _event_emitter is None:
        _event_emitter = EventEmitter()
    return _event_emitter


def get_shared_state(persist: bool = False) -> SharedState:
    """Get or create global SharedState instance."""
    global _shared_state
    if _shared_state is None:
        _shared_state = SharedState(persist=persist)
    return _shared_state


def get_task_queue(persist: bool = False) -> TaskQueue:
    """Get or create global TaskQueue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue(persist=persist)
    return _task_queue


def get_result_cache(
    max_size: int = 1000,
    default_ttl: int = 3600,
) -> ResultCache:
    """Get or create global ResultCache instance."""
    global _result_cache
    if _result_cache is None:
        _result_cache = ResultCache(max_size=max_size, default_ttl=default_ttl)
    return _result_cache


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point for agent communication."""
    import sys

    print("\n" + "=" * 60)
    print("  AGENT COMMUNICATION SYSTEM")
    print("=" * 60 + "\n")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "bus":
            bus = get_message_bus()
            stats = bus.get_stats()
            print(f"Message Bus Statistics:")
            print(f"  Topics: {', '.join(stats['topics']) or 'None'}")
            print(f"  Subscriptions: {stats['total_subscriptions']}")
            print(f"  Messages Processed: {stats['messages_processed']}")
            print(f"  Dead Letters: {stats['dead_letters']}")

        elif command == "events":
            emitter = get_event_emitter()
            stats = emitter.get_stats()
            print(f"Event Emitter Statistics:")
            print(f"  Event Types: {', '.join(stats['event_types']) or 'None'}")
            print(f"  Listeners: {stats['total_listeners']}")
            print(f"  History Size: {stats['history_size']}")

        elif command == "state":
            state = get_shared_state()
            stats = state.get_stats()
            print(f"Shared State Statistics:")
            print(f"  Default Keys: {stats['default_keys']}")
            print(f"  Namespaces: {', '.join(stats['namespaces']) or 'None'}")
            print(f"  Watchers: {stats['watchers']}")

        elif command == "queue":
            queue = get_task_queue()
            stats = queue.get_stats()
            print(f"Task Queue Statistics:")
            print(f"  Pending: {stats['pending']}")
            print(f"  Processing: {stats['processing']}")
            print(f"  Completed: {stats['completed']}")
            print(f"  Queue Size: {stats['queue_size']}")

        elif command == "cache":
            cache = get_result_cache()
            stats = cache.get_stats()
            print(f"Result Cache Statistics:")
            print(f"  Size: {stats['size']}/{stats['max_size']}")
            print(f"  Hits: {stats['hits']}")
            print(f"  Misses: {stats['misses']}")
            print(f"  Hit Rate: {stats['hit_rate']:.1%}")

        elif command == "all":
            # Show all statistics
            print("Message Bus:")
            for k, v in get_message_bus().get_stats().items():
                print(f"  {k}: {v}")
            print("\nEvent Emitter:")
            for k, v in get_event_emitter().get_stats().items():
                print(f"  {k}: {v}")
            print("\nShared State:")
            for k, v in get_shared_state().get_stats().items():
                print(f"  {k}: {v}")
            print("\nTask Queue:")
            for k, v in get_task_queue().get_stats().items():
                print(f"  {k}: {v}")
            print("\nResult Cache:")
            for k, v in get_result_cache().get_stats().items():
                print(f"  {k}: {v}")

    else:
        print("Usage:")
        print("  python -m src.agents.agent_communication bus    # Message bus stats")
        print("  python -m src.agents.agent_communication events # Event emitter stats")
        print("  python -m src.agents.agent_communication state  # Shared state stats")
        print("  python -m src.agents.agent_communication queue  # Task queue stats")
        print("  python -m src.agents.agent_communication cache  # Result cache stats")
        print("  python -m src.agents.agent_communication all    # All statistics")


if __name__ == "__main__":
    main()
