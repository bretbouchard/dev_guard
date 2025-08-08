"""
System resilience and error recovery tests for DevGuard.
Tests system stability under failure conditions and error recovery mechanisms.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from dev_guard.core.config import Config
from dev_guard.core.swarm import DevGuardSwarm
from dev_guard.memory.shared_memory import SharedMemory
from dev_guard.memory.vector_db import VectorDatabase


class TestSystemResilience:
    """Test suite for system resilience and error recovery."""

    @pytest.fixture
    async def resilience_environment(self, tmp_path):
        """Set up test environment for resilience testing."""
        config_data = {
            "llm": {
                "provider": "openrouter",
                "model": "qwen/qwen-2.5-coder-32b-instruct",
                "api_key": "test-key",
                "max_retries": 3,
                "timeout": 10
            },
            "shared_memory": {
                "provider": "sqlite",
                "db_path": str(tmp_path / "test_memory.db")
            },
            "vector_db": {
                "provider": "chroma",
                "path": str(tmp_path / "vector_db")
            },
            "agents": {
                "commander": {"enabled": True, "max_retries": 2},
                "planner": {"enabled": True, "max_retries": 2},
                "code": {"enabled": True, "max_retries": 2}
            }
        }
        
        config = Config.load_from_dict(config_data)
        shared_memory = SharedMemory(db_path=str(tmp_path / "test_memory.db"))
        vector_db = VectorDatabase(config.vector_db)
        
        return {
            "config": config,
            "shared_memory": shared_memory,
            "vector_db": vector_db,
            "tmp_path": tmp_path
        }

    @pytest.mark.asyncio
    async def test_llm_provider_failure_recovery(self, resilience_environment):
        """Test system recovery from LLM provider failures."""
        env = resilience_environment
        
        # Mock LLM client with intermittent failures
        mock_llm = AsyncMock()
        failure_count = 0
        
        async def failing_completion(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError("LLM provider temporarily unavailable")
            return AsyncMock(
                content='{"success": true, "message": "recovered"}'
            )
        
        mock_llm.chat_completion.side_effect = failing_completion
        
        with patch(
            'dev_guard.core.swarm.OpenRouterClient', 
            return_value=mock_llm
        ):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Submit request that should recover after failures
            request = {
                "type": "test_request",
                "description": "Test LLM failure recovery"
            }
            
            start_time = time.time()
            result = await swarm.process_user_request(request)
            end_time = time.time()
            
            # Verify recovery succeeded
            assert result["success"], "Failed to recover from LLM failures"
            assert failure_count == 3, (
                f"Expected 3 attempts, got {failure_count}"
            )
            assert end_time - start_time > 2, (
                "Recovery too fast, retries not working"
            )
            
            # Verify error recovery was logged
            error_entries = env["shared_memory"].search_entries(
                tags={"error", "recovery"}
            )
            assert len(error_entries) > 0, "Error recovery not logged"
            
            print("✅ LLM provider failure recovery test passed")

    @pytest.mark.asyncio
    async def test_database_connection_failure_recovery(self, resilience_environment):
        """Test recovery from database connection failures."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.chat_completion.return_value = AsyncMock(
                content='{"success": true}'
            )
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Simulate database connection failure
            original_add_memory = env["shared_memory"].add_memory
            call_count = 0
            
            def failing_add_memory(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError("Database connection lost")
                return original_add_memory(*args, **kwargs)
            
            env["shared_memory"].add_memory = failing_add_memory
            
            # Submit request that triggers database operations
            request = {
                "type": "database_test",
                "description": "Test database failure recovery"
            }
            
            result = await swarm.process_user_request(request)
            
            # Verify system continued operating despite database failures
            assert result["success"], "System failed to handle database errors"
            assert call_count == 3, f"Expected 3 database attempts, got {call_count}"
            
            print("✅ Database failure recovery test passed")

    @pytest.mark.asyncio
    async def test_agent_failure_and_fallback(self, resilience_environment):
        """Test agent failure handling and fallback mechanisms."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            # Mock primary agent failure, secondary success
            mock_llm = AsyncMock()
            responses = [
                # Primary agent fails
                AsyncMock(side_effect=RuntimeError("Agent crashed")),
                # Fallback agent succeeds
                AsyncMock(content='{"success": true, "agent": "fallback", "message": "Task completed by fallback agent"}')
            ]
            mock_llm.chat_completion.side_effect = responses
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            request = {
                "type": "agent_fallback_test",
                "description": "Test agent failure and fallback",
                "preferred_agent": "primary",
                "fallback_agents": ["secondary", "tertiary"]
            }
            
            result = await swarm.process_user_request(request)
            
            assert result["success"], "Fallback mechanism failed"
            
            # Verify fallback was logged
            fallback_entries = env["shared_memory"].search_entries(
                tags={"agent", "fallback"}
            )
            assert len(fallback_entries) > 0, "Agent fallback not logged"
            
            print("✅ Agent failure and fallback test passed")

    @pytest.mark.asyncio
    async def test_concurrent_load_stability(self, resilience_environment):
        """Test system stability under concurrent load."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            
            # Simulate varying response times
            async def variable_response(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                return AsyncMock(content='{"success": true, "timestamp": "' + str(time.time()) + '"}')
            
            mock_llm.chat_completion.side_effect = variable_response
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Submit multiple concurrent requests
            concurrent_requests = []
            for i in range(10):
                request = {
                    "type": "concurrent_test",
                    "description": f"Concurrent request {i}",
                    "request_id": i
                }
                concurrent_requests.append(
                    swarm.process_user_request(request)
                )
            
            # Execute all requests concurrently
            start_time = time.time()
            results = await asyncio.gather(*concurrent_requests, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception) and r.get("success")]
            failed_results = [r for r in results if isinstance(r, Exception) or not r.get("success")]
            
            # Verify system handled concurrent load
            assert len(successful_results) >= 8, f"Too many failures: {len(failed_results)}/10"
            assert end_time - start_time < 5, "Concurrent processing too slow"
            
            # Verify no data corruption in shared memory
            memory_entries = env["shared_memory"].get_recent_entries(limit=50)
            unique_timestamps = set()
            for entry in memory_entries:
                if hasattr(entry, 'timestamp'):
                    unique_timestamps.add(entry.timestamp)
            
            assert len(unique_timestamps) > 1, "Possible data corruption in timestamps"
            
            print(f"✅ Concurrent load test passed: {len(successful_results)}/10 successful")

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, resilience_environment):
        """Test system behavior under memory pressure."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.chat_completion.return_value = AsyncMock(
                content='{"success": true, "data": "test_data"}'
            )
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Generate high memory usage by creating many memory entries
            large_data = "x" * 10000  # 10KB of data per entry
            
            for i in range(100):  # Create 100 entries (~1MB total)
                request = {
                    "type": "memory_pressure_test",
                    "description": "Large data processing",
                    "data": large_data,
                    "iteration": i
                }
                
                result = await swarm.process_user_request(request)
                assert result["success"], f"Failed at iteration {i}"
                
                # Check memory cleanup periodically
                if i % 20 == 19:
                    memory_count = len(env["shared_memory"].get_recent_entries(limit=1000))
                    assert memory_count < 500, f"Memory not being cleaned up: {memory_count} entries"
            
            # Verify system still responsive after memory pressure
            final_request = {
                "type": "final_test",
                "description": "Test system responsiveness after memory pressure"
            }
            
            final_result = await swarm.process_user_request(final_request)
            assert final_result["success"], "System not responsive after memory pressure"
            
            print("✅ Memory pressure handling test passed")

    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self, resilience_environment):
        """Test recovery from network timeouts."""
        env = resilience_environment
        
        timeout_count = 0
        
        async def timeout_then_succeed(*args, **kwargs):
            nonlocal timeout_count
            timeout_count += 1
            
            if timeout_count <= 2:
                await asyncio.sleep(15)  # Simulate timeout (longer than 10s config)
                raise TimeoutError("Network timeout")
            
            return AsyncMock(content='{"success": true, "recovered": true}')
        
        mock_llm = AsyncMock()
        mock_llm.chat_completion.side_effect = timeout_then_succeed
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=mock_llm):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            request = {
                "type": "timeout_recovery_test",
                "description": "Test network timeout recovery"
            }
            
            start_time = time.time()
            result = await swarm.process_user_request(request)
            end_time = time.time()
            
            assert result["success"], "Failed to recover from timeouts"
            assert timeout_count == 3, f"Expected 3 timeout attempts, got {timeout_count}"
            # Should take less than 45s (3 * 15s) due to timeout handling
            assert end_time - start_time < 40, "Timeout recovery took too long"
            
            # Verify timeout recovery was logged
            timeout_entries = env["shared_memory"].search_entries(
                tags={"timeout", "recovery"}
            )
            assert len(timeout_entries) > 0, "Timeout recovery not logged"
            
            print("✅ Network timeout recovery test passed")

    @pytest.mark.asyncio
    async def test_data_consistency_during_failures(self, resilience_environment):
        """Test data consistency is maintained during system failures."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock responses with some failures
            responses = [
                AsyncMock(content='{"success": true, "step": 1, "data": "initial"}'),
                AsyncMock(side_effect=RuntimeError("Failure during processing")),
                AsyncMock(content='{"success": true, "step": 3, "data": "recovered"}')
            ]
            mock_llm.chat_completion.side_effect = responses
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Submit multi-step request that has intermediate failure
            request = {
                "type": "data_consistency_test",
                "description": "Multi-step process with intermediate failure",
                "steps": ["initialize", "process", "finalize"],
                "transaction_id": "test_transaction_123"
            }
            
            result = await swarm.process_user_request(request)
            
            # Verify final result
            assert result["success"], "Transaction failed to complete"
            
            # Verify data consistency in shared memory
            transaction_entries = env["shared_memory"].search_entries(
                content_filter={"transaction_id": "test_transaction_123"}
            )
            
            # Should have entries for successful steps and error handling
            assert len(transaction_entries) >= 2, "Transaction not properly logged"
            
            # Verify no partial data corruption
            for entry in transaction_entries:
                assert entry.content is not None, "Corrupted entry found"
                if "transaction_id" in str(entry.content):
                    assert "test_transaction_123" in str(entry.content), "Transaction ID corrupted"
            
            print("✅ Data consistency during failures test passed")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_and_restart(self, resilience_environment):
        """Test graceful system shutdown and restart capabilities."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.chat_completion.return_value = AsyncMock(
                content='{"success": true, "status": "running"}'
            )
            mock_llm_class.return_value = mock_llm
            
            # Initialize and start system
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Submit some requests to create state
            for i in range(5):
                request = {
                    "type": "pre_shutdown_test",
                    "description": f"Request {i} before shutdown",
                    "request_id": f"pre_shutdown_{i}"
                }
                result = await swarm.process_user_request(request)
                assert result["success"], f"Pre-shutdown request {i} failed"
            
            # Verify system state before shutdown
            pre_shutdown_entries = env["shared_memory"].get_recent_entries(limit=10)
            assert len(pre_shutdown_entries) >= 5, "Insufficient pre-shutdown state"
            
            # Perform graceful shutdown
            await swarm.shutdown(graceful=True)
            
            # Verify system stopped
            assert not swarm.is_running, "System not properly stopped"
            
            # Restart system
            new_swarm = DevGuardSwarm(env["config"])
            await new_swarm.initialize()
            
            # Verify state recovery
            post_restart_entries = env["shared_memory"].get_recent_entries(limit=20)
            assert len(post_restart_entries) >= len(pre_shutdown_entries), "State not recovered after restart"
            
            # Submit post-restart request
            post_restart_request = {
                "type": "post_restart_test",
                "description": "Test after restart",
                "request_id": "post_restart"
            }
            
            result = await new_swarm.process_user_request(post_restart_request)
            assert result["success"], "System not functional after restart"
            
            print("✅ Graceful shutdown and restart test passed")

    @pytest.mark.asyncio
    async def test_partial_system_failure_resilience(self, resilience_environment):
        """Test system resilience when partial components fail."""
        env = resilience_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock vector database failure
            original_vector_search = env["vector_db"].search
            
            def failing_vector_search(*args, **kwargs):
                raise ConnectionError("Vector database unavailable")
            
            env["vector_db"].search = failing_vector_search
            
            # Mock successful LLM but failed vector search
            mock_llm.chat_completion.return_value = AsyncMock(
                content='{"success": true, "fallback_mode": true}'
            )
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Submit request that would normally use vector database
            request = {
                "type": "knowledge_search_test",
                "description": "Test with vector database failure",
                "requires_context": True
            }
            
            result = await swarm.process_user_request(request)
            
            # System should still function without vector database
            assert result["success"], "System failed when vector DB unavailable"
            
            # Verify fallback mode was logged
            fallback_entries = env["shared_memory"].search_entries(
                tags={"fallback", "vector_db"}
            )
            assert len(fallback_entries) > 0, "Vector DB fallback not logged"
            
            # Restore vector database and test recovery
            env["vector_db"].search = original_vector_search
            
            recovery_request = {
                "type": "recovery_test",
                "description": "Test vector database recovery"
            }
            
            recovery_result = await swarm.process_user_request(recovery_request)
            assert recovery_result["success"], "System didn't recover when vector DB restored"
            
            print("✅ Partial system failure resilience test passed")
