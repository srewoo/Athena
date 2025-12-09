"""
WebSocket Manager for Athena
Handles real-time updates for evaluation progress
"""

from fastapi import WebSocket
from typing import Dict, Set
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        # Map of evaluation_id to set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Keep track of all connections
        self.all_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket, evaluation_id: str = None):
        """
        Accept and store a new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            evaluation_id: Optional evaluation ID to subscribe to
        """
        await websocket.accept()
        self.all_connections.add(websocket)
        
        if evaluation_id:
            if evaluation_id not in self.active_connections:
                self.active_connections[evaluation_id] = set()
            self.active_connections[evaluation_id].add(websocket)
            
            logger.info(f"✅ WebSocket connected for evaluation: {evaluation_id}")
            
            # Send connection confirmation
            await websocket.send_json({
                "type": "connected",
                "evaluation_id": evaluation_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            logger.info("✅ WebSocket connected (no specific evaluation)")
    
    async def disconnect(self, websocket: WebSocket, evaluation_id: str = None):
        """
        Remove a WebSocket connection
        
        Args:
            websocket: WebSocket connection
            evaluation_id: Optional evaluation ID
        """
        self.all_connections.discard(websocket)
        
        if evaluation_id and evaluation_id in self.active_connections:
            self.active_connections[evaluation_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[evaluation_id]:
                del self.active_connections[evaluation_id]
            
            logger.info(f"❌ WebSocket disconnected for evaluation: {evaluation_id}")
        else:
            logger.info("❌ WebSocket disconnected")
    
    async def send_progress(self, evaluation_id: str, progress_data: dict):
        """
        Send progress update to all connections subscribed to an evaluation
        
        Args:
            evaluation_id: Evaluation ID
            progress_data: Progress data to send
        """
        if evaluation_id not in self.active_connections:
            return
        
        message = {
            "type": "progress",
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat(),
            **progress_data
        }
        
        # Send to all subscribers
        dead_connections = set()
        for connection in self.active_connections[evaluation_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error sending to WebSocket: {e}")
                dead_connections.add(connection)
        
        # Clean up dead connections
        for connection in dead_connections:
            await self.disconnect(connection, evaluation_id)
    
    async def send_completion(self, evaluation_id: str, result_data: dict):
        """
        Send completion notification
        
        Args:
            evaluation_id: Evaluation ID
            result_data: Final result data
        """
        if evaluation_id not in self.active_connections:
            return
        
        message = {
            "type": "completed",
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat(),
            **result_data
        }
        
        # Send to all subscribers
        connections = list(self.active_connections.get(evaluation_id, []))
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error sending completion: {e}")
    
    async def send_error(self, evaluation_id: str, error_message: str):
        """
        Send error notification
        
        Args:
            evaluation_id: Evaluation ID
            error_message: Error message
        """
        if evaluation_id not in self.active_connections:
            return
        
        message = {
            "type": "error",
            "evaluation_id": evaluation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message
        }
        
        # Send to all subscribers
        connections = list(self.active_connections.get(evaluation_id, []))
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error sending error notification: {e}")
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Message dictionary to broadcast
        """
        dead_connections = set()
        
        for connection in self.all_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"❌ Error broadcasting: {e}")
                dead_connections.add(connection)
        
        # Clean up dead connections
        for connection in dead_connections:
            self.all_connections.discard(connection)
    
    def get_connection_count(self, evaluation_id: str = None) -> int:
        """
        Get number of active connections
        
        Args:
            evaluation_id: Optional evaluation ID to filter by
            
        Returns:
            Number of connections
        """
        if evaluation_id:
            return len(self.active_connections.get(evaluation_id, set()))
        return len(self.all_connections)
    
    def get_stats(self) -> dict:
        """
        Get connection statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_connections": len(self.all_connections),
            "evaluation_subscriptions": len(self.active_connections),
            "evaluations": list(self.active_connections.keys())
        }


# Global connection manager instance
manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance"""
    return manager
