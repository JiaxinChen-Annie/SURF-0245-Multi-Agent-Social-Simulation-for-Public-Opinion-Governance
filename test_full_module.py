# test_full_module.py
# B 模块完整联调测试：模拟 A(环境) 和 C(LLM)，输出给 E 的最终数据

import json
import random
from pprint import pprint

from social_agent import (
    SocialAgent, AgentType, ActionType, 
    OpinionBelief, Perception, SocialInfo, MemoryRecord,
    BeliefSystem, EmotionState, Desire, Intention
)

# ============================================================
# 1. 模拟 A 模块（环境）和 C 模块（LLM 工具）
# ============================================================

class DummyLLMClient:
    """假 LLM：返回预定义的 JSON，模拟大模型决策"""
    def chat(self, prompt):
        # 你可以修改这个字典，来测试不同的 LLM 输出场景
        return json.dumps({
            "opinion_updates": {
                "E001": 0.85,     # 对事件1的观点变为 +0.85
                "E002": -0.30     # 对事件2的观点变为 -0.30
            },
            "emotion_delta": {
                "valence": 0.40,  # 情绪效价提升 0.4
                "arousal": -0.10  # 唤醒度降低 0.1
            }
        })

class DummyLLMUtils:
    def build_prompt(self, belief, memory, env_info):
        # 这里模拟构造 prompt，实际直接返回固定字符串即可
        print(f"   [联调] build_prompt 被调用，env_info 类型: {type(env_info)}")
        return "这是一个模拟的 Prompt"
    
    def get_client(self):
        return DummyLLMClient()
    
    def parse_llm_response(self, raw):
        return json.loads(raw)

class DummyModel:
    """模拟 A 模块的环境，负责接收 ActionRecord 并提供感知数据"""
    def __init__(self):
        self.random = random.Random(42)
        self.schedule = type('Obj', (), {'time': 10})()  # 当前仿真步数 tick=10
        self.llm_utils = DummyLLMUtils()
        self.trending = [("疫情政策", 0.95), ("经济复苏", 0.70)]
        
        # 用于捕获 B 提交的行动（最终要交给 E）
        self.last_action = None
    
    def submit_action(self, record):
        """B 调用此方法提交行动 -> 模拟写入环境缓存 -> 最终被 E 采集"""
        self.last_action = record
        print(f"\n   ✅ [环境-A] 收到 ActionRecord，已缓存等待 E 采集")
        print(f"      行动内容: {record.content[:30] if record.content else '(无内容)'}...")
        return None
    
    def get_neighbor_actions(self, agent_id):
        """模拟邻居的发言（供 B 感知）"""
        return [
            SocialInfo(source_id=2, action_type=ActionType.POST, 
                       content="邻居A：我支持这个政策", event_id="E001", opinion_value=0.70, timestamp=9),
            SocialInfo(source_id=3, action_type=ActionType.COMMENT, 
                       content="邻居B：我觉得有点问题", event_id="E001", opinion_value=-0.40, timestamp=9),
            SocialInfo(source_id=4, action_type=ActionType.REPOST, 
                       content="邻居C：转发官方消息", event_id="E002", opinion_value=0.10, timestamp=10),
        ]
    
    def get_mentions(self, agent_id):
        """模拟 @ 本智能体的消息"""
        return [
            SocialInfo(source_id=5, action_type=ActionType.COMMENT, 
                       content="@你 快来发表意见", event_id="E001", opinion_value=0.00, timestamp=10)
        ]

# ============================================================
# 2. 创建测试用智能体（预设初始信念）
# ============================================================

def create_test_agent():
    model = DummyModel()
    agent = SocialAgent(
        unique_id=1,
        model=model,
        agent_type=AgentType.PUBLIC,
        init_config={"role_desc": "普通网民", "stance_prior": 0.0}
    )
    # 预设两个事件的初始观点
    agent.beliefs.opinions["E001"] = OpinionBelief(event_id="E001", opinion_value=0.20, confidence=0.2)
    agent.beliefs.opinions["E002"] = OpinionBelief(event_id="E002", opinion_value=-0.10, confidence=0.40)
    # 预设初始情绪
    agent.beliefs.emotion.valence = -0.5
    agent.beliefs.emotion.arousal = 0.60
    # 清空旧记忆（模拟全新开始）
    agent.memory = []
    return agent

# ============================================================
# 3. 主流程：运行完整 step() 并输出给 E 的数据
# ============================================================

def main():
    print("\n" + "="*70)
    print("🧪 B 模块完整联调测试 (模拟 A 环境 + C LLM)")
    print("="*70)
    
    # ---------- 3.1 准备输入假数据 ----------
    print("\n📥 [输入] 假数据注入:")
    agent = create_test_agent()
    
    # 手动填充一些短期记忆（模拟历史交互）
    agent.memory = [
        MemoryRecord(tick=8, info=SocialInfo(source_id=2, action_type=ActionType.POST,
                                             content="之前聊过这事", event_id="E001", opinion_value=0.30, timestamp=8), relevance=0.80),
        MemoryRecord(tick=9, info=SocialInfo(source_id=1, action_type=ActionType.COMMENT,
                                             content="我之前说过一次", event_id="E001", opinion_value=0.40, timestamp=9), relevance=0.90),
    ]
    
    print(f"  - Agent ID: {agent.unique_id}, 类型: {agent.agent_type.name}")
    print(f"  - 初始观点 E001: {agent.beliefs.opinions['E001'].opinion_value}")
    print(f"  - 初始观点 E002: {agent.beliefs.opinions['E002'].opinion_value}")
    print(f"  - 初始情绪效价: {agent.beliefs.emotion.valence}")
    print(f"  - 短期记忆条数: {len(agent.memory)}")
    print(f"  - 环境热搜: {agent.model.trending}")
    
    # ---------- 3.2 执行完整推理 ----------
    print("\n⚙️ [处理] 执行 agent.step() 完整流程 (感知→信念→欲望→意图→行动)...")
    agent.step()
    
    # ---------- 3.3 输出给 E 的数据 ----------
    print("\n" + "="*70)
    print("📤 [输出给 E 模块的数据]")
    print("="*70)
    
    # ---- 输出 1：通过 A 的环境提交给 E 的 ActionRecord ----
    print("\n1️⃣  ActionRecord (通过 A.submit_action 提交 → 被 E 采集):")
    action = agent.model.last_action
    if action:
        print(f"   - agent_id      : {action.agent_id}")
        print(f"   - action_type   : {action.action_type.name}")
        print(f"   - content       : {action.content}")
        print(f"   - event_id      : {action.event_id}")
        print(f"   - opinion_value : {action.opinion_value}")
        print(f"   - target_id     : {action.target_id}")
        print(f"   - tick          : {action.tick}")
    else:
        print("   ⚠️ 本次行动为 SILENT（沉默），无 ActionRecord 产生")
    
    # ---- 输出 2：通过 A 的 DataCollector 采集的信念状态 ----
    print("\n2️⃣  BeliefSystem (被 A 的 DataCollector 回调读取 → 最终给 E 绘图):")
    print(f"   - 观点 E001  : {agent.beliefs.opinions['E001'].opinion_value:.3f} (置信度 {agent.beliefs.opinions['E001'].confidence:.2f})")
    print(f"   - 观点 E002  : {agent.beliefs.opinions['E002'].opinion_value:.3f} (置信度 {agent.beliefs.opinions['E002'].confidence:.2f})")
    print(f"   - 情绪效价   : {agent.beliefs.emotion.valence:.3f}")
    print(f"   - 情绪唤醒度 : {agent.beliefs.emotion.arousal:.3f}")
    print(f"   - 人格开放性 : {agent.beliefs.psychology.personality.openness:.3f}")
    print(f"   - 风险规避   : {agent.beliefs.psychology.risk_aversion:.3f}")
    
    # ---- 输出 3：更新后的记忆（辅助 E 做内容分析） ----
    print("\n3️⃣  短期记忆 (Memory, 最近3条):")
    for i, mem in enumerate(agent.memory[-3:], 1):
        print(f"   - [{i}] tick={mem.tick}, 内容={mem.info.content[:15]}..., 相关性={mem.relevance:.2f}")
    
    # ---------- 3.4 联调说明 ----------
    print("\n" + "="*70)
    print("✅ 联调检查清单:")
    print("  - ActionRecord 是否已被 A 缓存？", end=" ")
    print(f"✅ (last_action = {agent.model.last_action is not None})")
    print("  - E 可以从 A 的 DataCollector 读取 BeliefSystem 进行可视化。")
    print("="*70)

if __name__ == "__main__":
    main()