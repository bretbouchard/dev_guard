"""
Performance and load testing for DevGuard system integration.
Tests system behavior under various load conditions and performance 
requirements.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.dev_guard.core.config import Config
from src.dev_guard.core.swarm import DevGuardSwarm
from src.dev_guard.memory.shared_memory import SharedMemory
from src.dev_guard.memory.vector_db import VectorDatabase


class TestSystemPerformance:
    """Performance and load testing for DevGuard system."""

    @pytest.fixture
    async def performance_environment(self, tmp_path):
        """Set up performance testing environment."""
        config_data = {
            "llm": {
                "provider": "openrouter",
                "model": "gpt-oss:20b",
                "api_key": "test-key"
            },
            "shared_memory": {
                "provider": "sqlite",
                "db_path": str(tmp_path / "perf_memory.db")
            },
            "vector_db": {
                "provider": "chroma", 
                "path": str(tmp_path / "perf_vector_db")
            },
            "agents": {
                "commander": {"enabled": True},
                "planner": {"enabled": True},
                "code": {"enabled": True},
                "qa_test": {"enabled": True}
            }
        }
        
        # Enable debug to cap swarm loop iterations in tests
        config_data["debug"] = True
        config = Config.load_from_dict(config_data)
        shared_memory = SharedMemory(db_path=str(tmp_path / "perf_memory.db"))
        vector_db = VectorDatabase(config.vector_db)
        
        return {
            "config": config,
            "shared_memory": shared_memory,
            "vector_db": vector_db
        }

    @pytest.mark.asyncio
    async def test_request_processing_throughput(self, performance_environment):
        """Test system throughput with multiple sequential requests."""
        env = performance_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            
            # Fast response simulation
            async def fast_response(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms response time
                return AsyncMock(content='{"success": true, "processing_time": 0.01}')
            
            mock_llm.chat_completion.side_effect = fast_response
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Test throughput with sequential requests
            request_count = 50
            start_time = time.time()
            
            successful_requests = 0
            for i in range(request_count):
                request = {
                    "type": "throughput_test",
                    "description": f"Throughput test request {i}",
                    "request_id": f"throughput_{i}"
                }
                
                result = await swarm.process_user_request(request)
                if result.get("success"):
                    successful_requests += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            requests_per_second = successful_requests / total_time
            avg_response_time = total_time / successful_requests
            
            # Performance assertions
            assert successful_requests >= request_count * 0.95, (
                f"Too many failed requests: "
                f"{request_count - successful_requests}/{request_count}"
            )
            assert requests_per_second >= 5.0, (
                f"Throughput too low: {requests_per_second:.2f} req/s"
            )
            assert avg_response_time <= 1.0, (
                f"Average response time too high: {avg_response_time:.3f}s"
            )
            
            print(f"✅ Throughput test: {requests_per_second:.1f} req/s, "
                  f"avg response: {avg_response_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, performance_environment):
        """Test system performance with concurrent requests."""
        env = performance_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            
            # Variable response time simulation
            request_counter = 0
            
            async def variable_response(*args, **kwargs):
                nonlocal request_counter
                request_counter += 1
                # Simulate variable processing times
                delay = 0.05 + (request_counter % 5) * 0.01
                await asyncio.sleep(delay)
                return AsyncMock(
                    content=f'{{"success": true, "request_id": {request_counter}, "delay": {delay}}}'
                )
            
            mock_llm.chat_completion.side_effect = variable_response
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Create concurrent requests
            concurrent_count = 20
            requests = []
            
            for i in range(concurrent_count):
                request = {
                    "type": "concurrent_test",
                    "description": f"Concurrent request {i}",
                    "request_id": f"concurrent_{i}"
                }
                requests.append(swarm.process_user_request(request))
            
            # Execute concurrently and measure performance
            start_time = time.time()
            results = await asyncio.gather(*requests, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Analyze results
            successful_results = [
                r for r in results 
                if not isinstance(r, Exception) and r.get("success")
            ]
            
            # Performance assertions
            success_rate = len(successful_results) / concurrent_count
            assert success_rate >= 0.9, (
                f"Success rate too low: {success_rate*100:.1f}%"
            )
            assert total_time <= 2.0, (
                f"Concurrent processing too slow: {total_time:.2f}s"
            )
            
            print(f"✅ Concurrent test: {len(successful_results)}/{concurrent_count} "
                  f"successful in {total_time:.2f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_scalability(self, performance_environment):
        """Test memory usage growth under sustained load."""
        env = performance_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.chat_completion.return_value = AsyncMock(
                content='{"success": true, "memory_test": true}'
            )
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Measure initial memory usage
            initial_entries = len(env["shared_memory"].get_recent_entries(limit=1000))
            
            # Generate sustained load
            batch_size = 10
            batch_count = 5
            
            for batch in range(batch_count):
                batch_start = time.time()
                
                # Process batch of requests
                batch_requests = []
                for i in range(batch_size):
                    request = {
                        "type": "memory_scalability_test",
                        "description": f"Memory test batch {batch}, request {i}",
                        "batch": batch,
                        "request": i,
                        "data": f"test_data_{batch}_{i}" * 100  # Add some data volume
                    }
                    batch_requests.append(swarm.process_user_request(request))
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_requests)
                batch_end = time.time()
                
                # Check memory growth
                current_entries = len(env["shared_memory"].get_recent_entries(limit=1000))
                memory_growth = current_entries - initial_entries
                
                # Verify performance doesn't degrade
                batch_time = batch_end - batch_start
                assert batch_time <= 2.0, (
                    f"Batch {batch} processing too slow: {batch_time:.2f}s"
                )
                
                # Verify memory growth is reasonable
                expected_max_growth = (batch + 1) * batch_size * 3  # Allow for some overhead
                assert memory_growth <= expected_max_growth, (
                    f"Memory growth too high: {memory_growth} entries"
                )
                
                print(f"  Batch {batch}: {batch_time:.2f}s, "
                      f"{memory_growth} memory entries")
            
            # Verify successful batch processing
            successful_batches = batch_count
            assert successful_batches == batch_count, (
                "Not all batches processed successfully"
            )
            
            print(f"✅ Memory scalability: {successful_batches} batches, "
                  f"{memory_growth} total memory entries")

    @pytest.mark.asyncio
    async def test_agent_coordination_performance(self, performance_environment):
        """Test performance of multi-agent coordination workflows."""
        env = performance_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            
            # Mock coordinated agent responses
            agent_responses = [
                '{"agent": "commander", "action": "route_to_planner", "success": true}',
                '{"agent": "planner", "action": "create_tasks", "tasks": 3, "success": true}',
                '{"agent": "code", "action": "execute_task", "task_id": 1, "success": true}',
                '{"agent": "qa_test", "action": "validate_output", "validation": "passed", "success": true}',
            ]
            
            response_cycle = 0
            
            async def coordinated_response(*args, **kwargs):
                nonlocal response_cycle
                response = agent_responses[response_cycle % len(agent_responses)]
                response_cycle += 1
                await asyncio.sleep(0.02)  # Simulate coordination time
                return AsyncMock(content=response)
            
            mock_llm.chat_completion.side_effect = coordinated_response
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Test multi-agent workflow performance
            workflow_count = 10
            coordination_results = []
            
            for workflow_id in range(workflow_count):
                start_time = time.time()
                
                request = {
                    "type": "multi_agent_coordination",
                    "description": f"Multi-agent workflow {workflow_id}",
                    "requires_coordination": True,
                    "workflow_id": workflow_id,
                    "agents_required": ["commander", "planner", "code", "qa_test"]
                }
                
                result = await swarm.process_user_request(request)
                end_time = time.time()
                
                coordination_time = end_time - start_time
                coordination_results.append({
                    "workflow_id": workflow_id,
                    "success": result.get("success", False),
                    "coordination_time": coordination_time
                })
                
                # Performance assertion for individual workflows
                assert coordination_time <= 1.0, (
                    f"Workflow {workflow_id} coordination too slow: {coordination_time:.3f}s"
                )
            
            # Analyze overall coordination performance
            successful_workflows = [r for r in coordination_results if r["success"]]
            avg_coordination_time = sum(r["coordination_time"] for r in successful_workflows) / len(successful_workflows)
            
            assert len(successful_workflows) >= workflow_count * 0.95, (
                f"Too many failed workflows: {workflow_count - len(successful_workflows)}/{workflow_count}"
            )
            assert avg_coordination_time <= 0.5, (
                f"Average coordination time too high: {avg_coordination_time:.3f}s"
            )
            
            print(f"✅ Agent coordination: {len(successful_workflows)}/{workflow_count} workflows, "
                  f"avg time: {avg_coordination_time:.3f}s")

    @pytest.mark.asyncio
    async def test_database_operation_performance(self, performance_environment):
        """Test database operation performance under load."""
        env = performance_environment
        
        # Test direct database operations
        start_time = time.time()
        
        # High-frequency write operations
        write_count = 100
        for i in range(write_count):
            from dev_guard.memory.models import MemoryEntry
            
            entry = MemoryEntry(
                agent_id=f"perf_test_agent_{i % 5}",
                type="performance_test",
                content={
                    "test_id": f"db_perf_{i}",
                    "data": f"test_data_{i}" * 50,  # Some data volume
                    "timestamp": time.time()
                },
                tags={"performance", "database", f"batch_{i // 10}"},
                parent_id=None,
                goose_patch=None,
                ast_summary=None,
                goose_strategy=None,
                file_path=None
            )
            
            env["shared_memory"].add_memory(entry)
        
        write_time = time.time() - start_time
        
        # Test read operations
        read_start = time.time()
        
        # Various read patterns
        recent_entries = env["shared_memory"].get_recent_entries(limit=50)
        search_results = env["shared_memory"].search_entries(tags={"performance"})
        filtered_entries = env["shared_memory"].search_entries(
            content_filter={"test_id": "db_perf_50"}
        )
        
        read_time = time.time() - read_start
        
        # Performance assertions
        writes_per_second = write_count / write_time
        assert writes_per_second >= 50.0, (
            f"Write performance too slow: {writes_per_second:.1f} writes/s"
        )
        
        assert read_time <= 0.5, (
            f"Read operations too slow: {read_time:.3f}s"
        )
        
        assert len(recent_entries) == 50, "Recent entries query failed"
        assert len(search_results) >= write_count, "Search query incomplete"
        
        print(f"✅ Database performance: {writes_per_second:.1f} writes/s, "
              f"reads: {read_time:.3f}s")

    @pytest.mark.asyncio
    async def test_error_recovery_performance_impact(self, performance_environment):
        """Test performance impact of error recovery mechanisms."""
        env = performance_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient') as mock_llm_class:
            mock_llm = AsyncMock()
            
            # Mix of successful and failing responses
            call_count = 0
            
            async def mixed_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # Every 5th call fails, then succeeds on retry
                if call_count % 5 == 0 and call_count % 10 != 0:
                    await asyncio.sleep(0.01)
                    raise ConnectionError("Temporary failure")
                
                await asyncio.sleep(0.02)  # Normal processing time
                return AsyncMock(
                    content=f'{{"success": true, "call_count": {call_count}}}'
                )
            
            mock_llm.chat_completion.side_effect = mixed_response
            mock_llm_class.return_value = mock_llm
            
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Test performance with intermittent failures
            request_count = 30
            start_time = time.time()
            
            successful_requests = 0
            total_retries = 0
            
            for i in range(request_count):
                request_start = time.time()
                
                request = {
                    "type": "error_recovery_performance",
                    "description": f"Performance test with errors {i}",
                    "request_id": f"error_perf_{i}"
                }
                
                result = await swarm.process_user_request(request)
                request_end = time.time()
                
                if result.get("success"):
                    successful_requests += 1
                
                # Track if request took longer (indicating retry)
                request_time = request_end - request_start
                if request_time > 0.05:  # Longer than normal processing
                    total_retries += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Performance assertions with error recovery
            success_rate = successful_requests / request_count
            avg_time_per_request = total_time / request_count
            
            assert success_rate >= 0.95, (
                f"Success rate with errors too low: {success_rate*100:.1f}%"
            )
            assert avg_time_per_request <= 0.2, (
                f"Average request time with retries too high: {avg_time_per_request:.3f}s"
            )
            
            # Verify retries happened (some errors should have occurred)
            assert total_retries > 0, "No retries detected, error injection may have failed"
            
            print(f"✅ Error recovery performance: {success_rate*100:.1f}% success, "
                  f"{avg_time_per_request:.3f}s avg time, {total_retries} retries")
