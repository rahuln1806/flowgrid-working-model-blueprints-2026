#!/usr/bin/env python3
"""
Redis Integration Module for Traffic Analysis
Sends detection results to Redis for real-time data sharing
"""

import json
import time
import argparse
from typing import Dict, Any, Optional
import redis
from redis.exceptions import ConnectionError, RedisError

class RedisIntegration:
    """Redis client for traffic detection data integration"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "cv:",
        default_key: str = "data",
        ttl: int = 60
    ):
        """
        Initialize Redis client
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            key_prefix: Prefix for all keys
            default_key: Default key for detection data
            ttl: Time-to-live for data in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.default_key = default_key
        self.ttl = ttl
        
        # Initialize Redis client
        self.client = None
        self._connect()
    
    def _connect(self) -> bool:
        """
        Connect to Redis server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.client.ping()
            print(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
            
        except ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
            self.client = None
            return False
        except Exception as e:
            print(f"❌ Redis connection error: {e}")
            self.client = None
            return False
    
    def is_connected(self) -> bool:
        """
        Check if Redis client is connected
        
        Returns:
            True if connected, False otherwise
        """
        if self.client is None:
            return False
        
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def send_detection_data(self, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Send detection data to Redis
        
        Args:
            data: Detection data dictionary
            key: Redis key (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return False
        
        try:
            # Use default key if none provided
            if key is None:
                key = self.default_key
            
            # Add full key with prefix
            full_key = f"{self.key_prefix}{key}"
            
            # Add timestamp to data
            data_with_timestamp = {
                "timestamp": int(time.time()),
                "data": data
            }
            
            # Convert to JSON and send
            json_data = json.dumps(data_with_timestamp)
            
            # Send to Redis with TTL
            self.client.setex(full_key, self.ttl, json_data)
            
            print(f"📤 Sent detection data to Redis: {full_key}")
            return True
            
        except RedisError as e:
            print(f"❌ Redis error sending data: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending data to Redis: {e}")
            return False
    
    def get_detection_data(self, key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detection data from Redis
        
        Args:
            key: Redis key (uses default if None)
            
        Returns:
            Detection data dictionary or None if not found
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return None
        
        try:
            # Use default key if none provided
            if key is None:
                key = self.default_key
            
            # Add full key with prefix
            full_key = f"{self.key_prefix}{key}"
            
            # Get data from Redis
            json_data = self.client.get(full_key)
            
            if json_data is None:
                print(f"⚠️ No data found for key: {full_key}")
                return None
            
            # Parse JSON
            data = json.loads(json_data)
            print(f"📥 Retrieved detection data from Redis: {full_key}")
            
            return data
            
        except RedisError as e:
            print(f"❌ Redis error getting data: {e}")
            return None
        except Exception as e:
            print(f"❌ Error getting data from Redis: {e}")
            return None
    
    def send_alert(self, alert_type: str, message: str, severity: str = "info") -> bool:
        """
        Send alert message to Redis
        
        Args:
            alert_type: Type of alert (e.g., "ambulance", "system")
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return False
        
        try:
            alert_data = {
                "type": "alert",
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": int(time.time())
            }
            
            # Use alerts key
            alert_key = f"{self.key_prefix}alerts"
            
            # Send as list (push to alerts list)
            self.client.lpush(alert_key, json.dumps(alert_data))
            
            # Keep only last 100 alerts
            self.client.ltrim(alert_key, 0, 99)
            
            # Set TTL for alerts list
            self.client.expire(alert_key, self.ttl * 10)  # Longer TTL for alerts
            
            print(f"🚨 Sent alert to Redis: {alert_type} - {message}")
            return True
            
        except RedisError as e:
            print(f"❌ Redis error sending alert: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending alert to Redis: {e}")
            return False
    
    def get_alerts(self, limit: int = 10) -> list:
        """
        Get recent alerts from Redis
        
        Args:
            limit: Maximum number of alerts to retrieve
            
        Returns:
            List of alert dictionaries
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return []
        
        try:
            alert_key = f"{self.key_prefix}alerts"
            
            # Get alerts from Redis list
            alert_jsons = self.client.lrange(alert_key, 0, limit - 1)
            
            alerts = []
            for alert_json in alert_jsons:
                try:
                    alert = json.loads(alert_json)
                    alerts.append(alert)
                except json.JSONDecodeError:
                    continue
            
            print(f"📥 Retrieved {len(alerts)} alerts from Redis")
            return alerts
            
        except RedisError as e:
            print(f"❌ Redis error getting alerts: {e}")
            return []
        except Exception as e:
            print(f"❌ Error getting alerts from Redis: {e}")
            return []
    
    def publish_detection(self, data: Dict[str, Any], channel: str = "detections") -> bool:
        """
        Publish detection data to Redis channel (pub/sub)
        
        Args:
            data: Detection data dictionary
            channel: Redis channel name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return False
        
        try:
            # Add timestamp
            data_with_timestamp = {
                "timestamp": int(time.time()),
                "data": data
            }
            
            # Convert to JSON and publish
            json_data = json.dumps(data_with_timestamp)
            
            # Publish to channel
            self.client.publish(channel, json_data)
            
            print(f"📡 Published detection data to channel: {channel}")
            return True
            
        except RedisError as e:
            print(f"❌ Redis error publishing data: {e}")
            return False
        except Exception as e:
            print(f"❌ Error publishing data to Redis: {e}")
            return False
    
    def set_system_status(self, status: str, details: Dict[str, Any] = None) -> bool:
        """
        Set system status in Redis
        
        Args:
            status: System status (active, inactive, error)
            details: Additional status details
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return False
        
        try:
            status_data = {
                "status": status,
                "timestamp": int(time.time()),
                "details": details or {}
            }
            
            status_key = f"{self.key_prefix}status"
            
            # Set status with longer TTL
            self.client.setex(status_key, self.ttl * 5, json.dumps(status_data))
            
            print(f"📊 Set system status: {status}")
            return True
            
        except RedisError as e:
            print(f"❌ Redis error setting status: {e}")
            return False
        except Exception as e:
            print(f"❌ Error setting status in Redis: {e}")
            return False
    
    def get_system_status(self) -> Optional[Dict[str, Any]]:
        """
        Get system status from Redis
        
        Returns:
            System status dictionary or None if not found
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return None
        
        try:
            status_key = f"{self.key_prefix}status"
            
            json_data = self.client.get(status_key)
            
            if json_data is None:
                return None
            
            return json.loads(json_data)
            
        except RedisError as e:
            print(f"❌ Redis error getting status: {e}")
            return None
        except Exception as e:
            print(f"❌ Error getting status from Redis: {e}")
            return None
    
    def clear_data(self, pattern: Optional[str] = None) -> bool:
        """
        Clear data from Redis
        
        Args:
            pattern: Key pattern to clear (uses prefix if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("❌ Not connected to Redis")
            return False
        
        try:
            if pattern is None:
                pattern = f"{self.key_prefix}*"
            
            # Get all matching keys
            keys = self.client.keys(pattern)
            
            if keys:
                # Delete all matching keys
                self.client.delete(*keys)
                print(f"🗑️ Cleared {len(keys)} keys matching pattern: {pattern}")
            else:
                print(f"⚠️ No keys found matching pattern: {pattern}")
            
            return True
            
        except RedisError as e:
            print(f"❌ Redis error clearing data: {e}")
            return False
        except Exception as e:
            print(f"❌ Error clearing data from Redis: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get Redis connection information
        
        Returns:
            Connection information dictionary
        """
        info = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "key_prefix": self.key_prefix,
            "connected": self.is_connected()
        }
        
        if self.client:
            try:
                # Get Redis server info
                server_info = self.client.info()
                info.update({
                    "redis_version": server_info.get("redis_version"),
                    "used_memory": server_info.get("used_memory_human"),
                    "connected_clients": server_info.get("connected_clients")
                })
            except:
                pass
        
        return info
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            print("🔌 Redis connection closed")

def main():
    """Main function for testing Redis integration"""
    parser = argparse.ArgumentParser(description="Test Redis integration for traffic detection")
    
    parser.add_argument("--host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--password", type=str, help="Redis password")
    parser.add_argument("--test", action="store_true", help="Run test operations")
    
    args = parser.parse_args()
    
    try:
        # Initialize Redis integration
        redis_client = RedisIntegration(
            host=args.host,
            port=args.port,
            password=args.password
        )
        
        if not redis_client.is_connected():
            print("❌ Cannot proceed without Redis connection")
            return 1
        
        if args.test:
            # Test sending detection data
            test_data = {
                "vehicles": 5,
                "pedestrians": 2,
                "ambulance": True,
                "total_objects": 7
            }
            
            print("🧪 Testing Redis integration...")
            
            # Send test data
            redis_client.send_detection_data(test_data)
            
            # Send test alert
            redis_client.send_alert("ambulance", "Ambulance detected in frame", "warning")
            
            # Set system status
            redis_client.set_system_status("active", {"fps": 25.5, "model": "yolov8n"})
            
            # Retrieve data
            retrieved_data = redis_client.get_detection_data()
            print(f"Retrieved data: {retrieved_data}")
            
            # Get alerts
            alerts = redis_client.get_alerts()
            print(f"Retrieved alerts: {alerts}")
            
            # Get connection info
            conn_info = redis_client.get_connection_info()
            print(f"Connection info: {conn_info}")
            
            print("✅ Redis integration test completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Redis integration test failed: {str(e)}")
        return 1
    finally:
        if 'redis_client' in locals():
            redis_client.close()

if __name__ == "__main__":
    exit(main())
