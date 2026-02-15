#!/usr/bin/env python3
"""
Multi-Agent Orchestration System
=================================
Implements the Brain and Muscles pattern from OpenClaw research.

Features:
- sessions_send: Inter-agent messaging
- sessions_spawn: Sub-agent creation
- agents_list: Available agents registry
- Task delegation with priority levels
- Memory flush protocol for context preservation

Based on OpenClaw Mission Control research.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import uuid4

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("multi_agent")

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class AgentProfile:
    """Profile for a specialized agent"""
    agent_id: str
    name: str
    role: str
    specialty: str
    capabilities: List[str]
    model: str = "minimax/MiniMax-M2.5"
    status: str = "available"
    current_task: str = None
    last_active: str = None


# ============================================================================
# AGENT REGISTRY
# ============================================================================
class AgentRegistry:
    """
    Registry of available agents and their capabilities.
    Used by the coordinator to select the right agent for each task.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
        self.message_queue: Dict[str, List[Dict]] = {}
        self._init_agents()
        
    def _init_agents(self):
        """Initialize default agent profiles"""
        default_agents = [
            AgentProfile(
                agent_id="coordinator",
                name="Eko Coordinator",
                role="Project Lead",
                specialty="Task breakdown and delegation",
                capabilities=["planning", "delegation", "monitoring", "reporting"]
            ),
            AgentProfile(
                agent_id="researcher",
                name="Eko Researcher",
                role="Researcher",
                specialty="Web search and data synthesis",
                capabilities=["web_search", "web_fetch", "sentiment_analysis", "news_aggregation"]
            ),
            AgentProfile(
                agent_id="devbot",
                name="Eko DevBot",
                role="Developer",
                specialty="Code writing and debugging",
                capabilities=["exec", "read", "write", "edit", "git", "debug"]
            ),
            AgentProfile(
                agent_id="auditor",
                name="Eko Auditor",
                role="Security Specialist",
                specialty="Code review and compliance",
                capabilities=["security_scan", "code_review", "compliance_check", "audit"]
            ),
            AgentProfile(
                agent_id="ux_manager",
                name="Eko UX Manager",
                role="Designer",
                specialty="Interface and visualization",
                capabilities=["dashboard", "visualization", "reports", "user_flow"]
            ),
            AgentProfile(
                agent_id="trading_agent",
                name="Eko Trader",
                role="Trading Specialist",
                specialty="Solana/Jupiter DEX",
                capabilities=["swap", "balance", "orders", "backtest"]
            ),
            AgentProfile(
                agent_id="risk_agent",
                name="Eko Risk Manager",
                role="Risk Specialist",
                specialty="Risk assessment",
                capabilities=["risk_assessment", "limit_check", "validation"]
            ),
            AgentProfile(
                agent_id="market_scanner",
                name="Eko Market Scanner",
                role="Market Scanner",
                specialty="Opportunity detection",
                capabilities=["volume_scan", "trend_detection", "pattern_recognition", "token_screening"]
            )
        ]
        
        for agent in default_agents:
            self.register(agent)
    
    def register(self, agent: AgentProfile):
        """Register a new agent"""
        self.agents[agent.agent_id] = agent
        self.message_queue[agent.agent_id] = []
        logger.info(f"✅ Agent registered: {agent.name} ({agent.agent_id})")
    
    def unregister(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.message_queue[agent_id]
            logger.info(f"❌ Agent unregistered: {agent_id}")
    
    def list_agents(self) -> List[Dict]:
        """List all registered agents"""
        return [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "role": a.role,
                "status": a.status,
                "capabilities": a.capabilities
            }
            for a in self.agents.values()
        ]
    
    def find_agent(self, capability: str) -> List[AgentProfile]:
        """Find agents with a specific capability"""
        return [
            a for a in self.agents.values()
            if capability in a.capabilities and a.status == "available"
        ]
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def update_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_active = datetime.now().isoformat()


# ============================================================================
# INTER-AGENT MESSAGING
# ============================================================================
class InterAgentMessaging:
    """
    Handles communication between agents using sessions_send pattern.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.message_history: List[Dict] = []
        
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        priority: str = "normal",
        context: Dict = None
    ) -> Dict:
        """
        Send message from one agent to another.
        
        Args:
            from_agent: Sender agent ID
            to_agent: Receiver agent ID
            message: Message content
            priority: normal, high, urgent
            context: Additional context data
        
        Returns:
            Message delivery receipt
        """
        message_id = str(uuid4())[:8]
        
        envelope = {
            "message_id": message_id,
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "priority": priority,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "status": "delivered"
        }
        
        # Queue message
        if to_agent in self.registry.message_queue:
            self.registry.message_queue[to_agent].append(envelope)
        
        # Update sender status
        self.registry.update_status(from_agent, f"sent to {to_agent}")
        
        # Update receiver status
        self.registry.update_status(to_agent, "has_message")
        
        # Log
        self.message_history.append(envelope)
        
        logger.info(f"📨 {from_agent} → {to_agent}: {message[:50]}...")
        
        return {
            "status": "delivered",
            "message_id": message_id,
            "to": to_agent
        }
    
    async def get_messages(self, agent_id: str) -> List[Dict]:
        """Get pending messages for an agent"""
        messages = self.registry.message_queue.get(agent_id, [])
        self.registry.message_queue[agent_id] = []
        return messages
    
    async def broadcast(
        self,
        from_agent: str,
        message: str,
        to_agents: List[str] = None
    ) -> List[Dict]:
        """Broadcast message to multiple agents"""
        targets = to_agents or list(self.registry.agents.keys())
        results = []
        
        for agent_id in targets:
            if agent_id != from_agent:
                result = await self.send_message(from_agent, agent_id, message)
                results.append(result)
        
        return results


# ============================================================================
# SUB-AGENT SPAWNER
# ============================================================================
class SubAgentSpawner:
    """
    Handles creation of sub-agents for parallel task execution.
    Implements sessions_spawn pattern from OpenClaw.
    """
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.active_subagents: Dict[str, Dict] = {}
        
    async def spawn(
        self,
        parent_agent: str,
        task: str,
        task_id: str = None,
        model: str = "minimax/MiniMax-M2.5-lightning"
    ) -> Dict:
        """
        Spawn a sub-agent for parallel task execution.
        
        Args:
            parent_agent: Agent creating the sub-agent
            task: Task description
            task_id: Optional specific task ID
            model: Model for the sub-agent
        
        Returns:
            Sub-agent session info
        """
        session_id = f"sub_{uuid4().hex[:12]}"
        sub_agent_id = f"{parent_agent}_sub_{len(self.active_subagents)}"
        
        subagent = {
            "session_id": session_id,
            "agent_id": sub_agent_id,
            "parent": parent_agent,
            "task": task,
            "model": model,
            "status": "running",
            "created_at": datetime.now().isoformat()
        }
        
        self.active_subagents[session_id] = subagent
        
        logger.info(f"🚀 {parent_agent} spawned {sub_agent_id} for: {task[:50]}...")
        
        return {
            "session_id": session_id,
            "agent_id": sub_agent_id,
            "status": "running"
        }
    
    async def terminate(self, session_id: str) -> Dict:
        """Terminate a sub-agent"""
        if session_id in self.active_subagents:
            self.active_subagents[session_id]["status"] = "terminated"
            self.active_subagents[session_id]["ended_at"] = datetime.now().isoformat()
            
            result = self.active_subagents[session_id].copy()
            del self.active_subagents[session_id]
            
            logger.info(f"🛑 Sub-agent {session_id} terminated")
            
            return result
        
        return {"error": "Session not found"}
    
    def list_active(self) -> List[Dict]:
        """List active sub-agents"""
        return list(self.active_subagents.values())


# ============================================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================================
class MultiAgentOrchestrator:
    """
    Main orchestrator implementing Brain and Muscles pattern.
    
    Coordinates multiple specialized agents for complex tasks.
    """
    
    def __init__(self):
        self.registry = AgentRegistry()
        self.messaging = InterAgentMessaging(self.registry)
        self.spawner = SubAgentSpawner(self.registry)
        self.task_history: List[Dict] = []
        self.memory_flush_enabled = True
        
        logger.info("🚀 Multi-Agent Orchestrator initialized")
    
    async def delegate_task(
        self,
        task: str,
        capability_required: str,
        priority: str = "normal"
    ) -> Dict:
        """
        Delegate a task to the best available agent.
        
        Args:
            task: Task description
            capability_required: Required capability
            priority: Task priority
        
        Returns:
            Task assignment result
        """
        # Find best agent
        agents = self.registry.find_agent(capability_required)
        
        if not agents:
            return {"error": f"No agent found with capability: {capability_required}"}
        
        # Select first available
        agent = agents[0]
        
        # Send task
        result = await self.messaging.send_message(
            from_agent="coordinator",
            to_agent=agent.agent_id,
            message=task,
            priority=priority,
            context={"capability": capability_required}
        )
        
        # Update agent status
        self.registry.update_status(agent.agent_id, f"working on: {task[:30]}...")
        
        # Record
        task_id = str(uuid4())[:8]
        self.task_history.append({
            "task_id": task_id,
            "task": task,
            "agent": agent.agent_id,
            "capability": capability_required,
            "status": "assigned",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "task_id": task_id,
            "agent": agent.agent_id,
            "agent_name": agent.name,
            "status": "assigned"
        }
    
    async def execute_workflow(self, workflow: Dict) -> Dict:
        """
        Execute a multi-step workflow with multiple agents.
        
        Workflow structure:
        {
            "name": "Deploy new strategy",
            "steps": [
                {"task": "Research SOL patterns", "agent": "researcher"},
                {"task": "Implement RSI", "agent": "devbot"},
                {"task": "Security review", "agent": "auditor"},
                {"task": "Create dashboard", "agent": "ux_manager"}
            ]
        }
        """
        results = []
        
        for step in workflow.get("steps", []):
            result = await self.delegate_task(
                task=step["task"],
                capability_required=step.get("agent", "general"),
                priority=step.get("priority", "normal")
            )
            results.append(result)
        
        return {
            "workflow": workflow.get("name"),
            "steps_completed": len(results),
            "results": results
        }
    
    async def memory_flush(self) -> Dict:
        """
        Perform memory flush before context compaction.
        Critical for preventing context loss.
        """
        flush_report = {
            "timestamp": datetime.now().isoformat(),
            "pending_tasks": len(self.task_history),
            "active_agents": [a for a in self.registry.agents.values() if a.status != "available"],
            "recent_decisions": self.task_history[-10:],
            "sub_agents_active": len(self.spawner.list_active())
        }
        
        # Save to persistent storage
        with open("data/multi_agent_memory.json", 'w') as f:
            json.dump(flush_report, f, indent=2, default=str)
        
        logger.info(f"💾 Memory flush completed: {len(self.task_history)} tasks recorded")
        
        return flush_report
    
    def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            "agents_registered": len(self.registry.agents),
            "agents_available": len(self.registry.find_agent("general")),
            "active_subagents": len(self.spawner.list_active()),
            "tasks_completed": len(self.task_history),
            "messages_sent": len(self.messaging.message_history)
        }
    
    def list_agents(self) -> List[Dict]:
        """List all registered agents"""
        return self.registry.list_agents()


# ============================================================================
# MAIN DEMO
# ============================================================================
async def main():
    """Demo multi-agent orchestration"""
    
    print("="*80)
    print("🚀 MULTI-AGENT ORCHESTRATION - Brain and Muscles Pattern")
    print("="*80)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # List agents
    print("\n📋 Registered Agents:")
    agents = orchestrator.list_agents()
    for agent in agents:
        print(f"   ✅ {agent['name']} ({agent['agent_id']})")
        print(f"      Role: {agent['role']}")
        print(f"      Capabilities: {', '.join(agent['capabilities'][:3])}")
    
    # Demo delegation
    print("\n📝 Demo Task Delegation:")
    
    tasks = [
        ("Research SOL trends", "web_search"),
        ("Implement RSI strategy", "write"),
        ("Security review", "security_scan"),
        ("Create dashboard", "dashboard")
    ]
    
    for task, capability in tasks:
        result = await orchestrator.delegate_task(task, capability)
        print(f"   ✅ {result.get('agent_name', result.get('error'))}: {task[:40]}")
    
    # Demo workflow
    print("\n🔄 Demo Multi-Step Workflow:")
    
    workflow = {
        "name": "Deploy New Trading Strategy",
        "steps": [
            {"task": "Research market conditions", "agent": "web_search", "priority": "high"},
            {"task": "Implement strategy code", "agent": "write", "priority": "normal"},
            {"task": "Security audit", "agent": "security_scan", "priority": "high"},
            {"task": "Create monitoring dashboard", "agent": "dashboard", "priority": "normal"}
        ]
    }
    
    workflow_result = await orchestrator.execute_workflow(workflow)
    print(f"   Workflow: {workflow_result['workflow']}")
    print(f"   Steps: {workflow_result['steps_completed']}")
    
    # Status
    print("\n📊 Orchestrator Status:")
    status = orchestrator.get_status()
    print(f"   Agents: {status['agents_registered']}")
    print(f"   Tasks: {status['tasks_completed']}")
    print(f"   Messages: {status['messages_sent']}")
    
    # Memory flush
    print("\n💾 Memory Flush Protocol:")
    flush = await orchestrator.memory_flush()
    print(f"   Tasks recorded: {flush['pending_tasks']}")
    print(f"   Saved to: data/multi_agent_memory.json")
    
    print("\n" + "="*80)
    print("✅ Multi-Agent Orchestration Demo Complete")
    print("="*80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
