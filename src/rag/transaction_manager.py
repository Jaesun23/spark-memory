"""Redis transaction manager.

This module handles Redis transaction processing and error recovery.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from src.redis.client import RedisClient

logger = logging.getLogger(__name__)


class RedisTransactionManager:
    """Redis transaction manager."""
    
    def __init__(self, redis_client: RedisClient):
        """Initialize.
        
        Args:
            redis_client: Redis client
        """
        self.redis = redis_client
        self.transaction_log: List[Dict[str, Any]] = []
        
    async def execute_with_transaction(
        self, 
        operations: List[Callable],
        rollback_operations: Optional[List[Callable]] = None,
        max_retries: int = 3
    ) -> bool:
        """Execute operations with transaction.
        
        Args:
            operations: List of operations to execute
            rollback_operations: List of rollback operations
            max_retries: Maximum retry count
            
        Returns:
            Success status
        """
        for attempt in range(max_retries):
            try:
                # Start transaction
                transaction_id = self._generate_transaction_id()
                self._log_transaction_start(transaction_id)
                
                # Start pipeline
                pipeline = await self.redis.pipeline()
                
                # Execute all operations
                for operation in operations:
                    await operation(pipeline)
                
                # Execute transaction
                results = await pipeline.execute()
                
                # Validate results
                if self._validate_results(results):
                    self._log_transaction_success(transaction_id)
                    return True
                else:
                    raise Exception("Transaction validation failed")
                    
            except Exception as e:
                logger.error(f"Transaction failed (attempt {attempt + 1}/{max_retries}): {e}")
                self._log_transaction_failure(transaction_id, str(e))
                
                # Execute rollback
                if rollback_operations:
                    await self._execute_rollback(rollback_operations)
                
                # Wait for retry if not last attempt
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        return False
    
    async def _execute_rollback(self, rollback_operations: List[Callable]) -> None:
        """Execute rollback operations.
        
        Args:
            rollback_operations: List of rollback operations
        """
        try:
            logger.info("Executing rollback operations")
            pipeline = await self.redis.pipeline()
            
            for operation in rollback_operations:
                await operation(pipeline)
                
            await pipeline.execute()
            logger.info("Rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    def _generate_transaction_id(self) -> str:
        """Generate transaction ID."""
        timestamp = datetime.now().timestamp()
        return f"txn_{int(timestamp * 1000000)}"
    
    def _log_transaction_start(self, transaction_id: str) -> None:
        """Log transaction start."""
        self.transaction_log.append({
            "id": transaction_id,
            "status": "started",
            "timestamp": datetime.now().isoformat(),
        })
    
    def _log_transaction_success(self, transaction_id: str) -> None:
        """Log transaction success."""
        self.transaction_log.append({
            "id": transaction_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
        })
    
    def _log_transaction_failure(self, transaction_id: str, error: str) -> None:
        """Log transaction failure."""
        self.transaction_log.append({
            "id": transaction_id,
            "status": "failed",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
    
    def _validate_results(self, results: List[Any]) -> bool:
        """Validate transaction results.
        
        Args:
            results: Pipeline execution results
            
        Returns:
            Whether all results are valid
        """
        # Check that all results are not None and have no errors
        for result in results:
            if isinstance(result, Exception):
                return False
        return True
    
    async def save_transaction_log(self) -> None:
        """Save transaction log to Redis."""
        if not self.transaction_log:
            return
            
        try:
            log_key = f"transaction:log:{datetime.now().strftime('%Y%m%d')}"
            
            # Save log as JSON
            await self.redis.client.json().set(
                log_key, 
                "$", 
                self.transaction_log
            )
            
            # Set TTL (7 days)
            await self.redis.client.expire(log_key, 7 * 24 * 60 * 60)
            
            logger.info(f"Saved {len(self.transaction_log)} transaction logs")
            
        except Exception as e:
            logger.error(f"Failed to save transaction log: {e}")


class RelationshipTransactionHelper:
    """Transaction helper for relationship storage."""
    
    @staticmethod
    def create_save_operations(
        chunk_id: str,
        relations: List[Any],
        key_patterns: Dict[str, str]
    ) -> List[Callable]:
        """Create relationship save operations.
        
        Args:
            chunk_id: Chunk ID
            relations: Relationship list
            key_patterns: Key pattern dictionary
            
        Returns:
            List of operation functions
        """
        operations = []
        
        # Metadata save operation
        async def save_metadata(pipeline):
            metadata_key = key_patterns["metadata"].format(chunk_id=chunk_id)
            metadata = {
                "total_relations": len(relations),
                "updated_at": datetime.now().isoformat(),
            }
            await pipeline.hset_dict(metadata_key, metadata)
        
        operations.append(save_metadata)
        
        # Per-relationship save operations
        for relation in relations:
            async def save_relation(pipeline, rel=relation):
                # Add target to Set
                set_key = key_patterns["set"].format(
                    type=rel.relation_type.value,
                    chunk_id=chunk_id
                )
                await pipeline.sadd(set_key, rel.target_id)
                
                # Save detailed information
                detail_key = key_patterns["detail"].format(
                    chunk_id=chunk_id,
                    target_id=rel.target_id
                )
                # Serialize relationship
                detail_data = {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type.value,
                    "strength": rel.strength,
                    "created_at": rel.created_at.isoformat(),
                    "metadata": rel.metadata,
                }
                await pipeline.json_set(detail_key, "$", detail_data)
            
            operations.append(save_relation)
        
        return operations
    
    @staticmethod
    def create_rollback_operations(
        chunk_id: str,
        relations: List[Any],
        key_patterns: Dict[str, str]
    ) -> List[Callable]:
        """Create rollback operations.
        
        Args:
            chunk_id: Chunk ID
            relations: Relationship list
            key_patterns: Key pattern dictionary
            
        Returns:
            List of rollback operation functions
        """
        operations = []
        
        # Delete metadata
        async def delete_metadata(pipeline):
            metadata_key = key_patterns["metadata"].format(chunk_id=chunk_id)
            await pipeline.delete(metadata_key)
        
        operations.append(delete_metadata)
        
        # Delete relationship data
        for relation in relations:
            async def delete_relation(pipeline, rel=relation):
                # Remove from Set
                set_key = key_patterns["set"].format(
                    type=rel.relation_type.value,
                    chunk_id=chunk_id
                )
                await pipeline.srem(set_key, rel.target_id)
                
                # Delete detailed information
                detail_key = key_patterns["detail"].format(
                    chunk_id=chunk_id,
                    target_id=rel.target_id
                )
                await pipeline.delete(detail_key)
            
            operations.append(delete_relation)
        
        return operations