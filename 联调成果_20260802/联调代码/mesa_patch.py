"""
mesa_patch.py — Mesa 3.x 兼容性补丁（在所有其他 import 之前运行）
-----------------------------------------------------------------
用法：在 opinion_model.py / test_module_a.py / run_sim.py 顶部第一行加：
    import mesa_patch  # noqa: F401  — 必须在所有 mesa import 之前

作用：将 Mesa 3.x 的 AgentSet 透明替换为 list，
      使 schedule.agents[0]、random.sample(agents)、len(agents) 全部正常工作。
"""

from __future__ import annotations

def _patch_mesa():
    try:
        import mesa
        from packaging import version
        mesa_ver = version.parse(mesa.__version__)
        is_v3 = mesa_ver >= version.parse("3.0")
    except Exception:
        # 无法判断版本或无 packaging，直接尝试补丁
        is_v3 = True

    if not is_v3:
        return  # Mesa 2.x 不需要补丁

    try:
        # ── 补丁1：RandomActivation.agents 返回 list ──────────────────
        try:
            from mesa.time import RandomActivation

            if not hasattr(RandomActivation, '_patched_agents'):
                _orig_agents = RandomActivation.__dict__.get('agents')

                @property  # type: ignore
                def _agents_as_list(self):
                    result = self.__dict__.get('_agents_list_cache')
                    # 每次都重新获取，确保最新
                    try:
                        raw = object.__getattribute__(self, '_mesa3_agents')
                        return list(raw)
                    except AttributeError:
                        pass
                    # 尝试各种可能的内部属性
                    for attr in ('_agents', 'agents_dict', '_agent_set'):
                        val = self.__dict__.get(attr)
                        if isinstance(val, dict):
                            return list(val.values())
                        if val is not None:
                            try:
                                return list(val)
                            except Exception:
                                pass
                    # 最后尝试原始属性
                    if _orig_agents is not None:
                        try:
                            raw = _orig_agents.fget(self)
                            return list(raw)
                        except Exception:
                            pass
                    return []

                RandomActivation.agents = _agents_as_list
                RandomActivation._patched_agents = True
                print("[mesa_patch] ✅ RandomActivation.agents → list 补丁已应用")

        except ImportError:
            pass

        # ── 补丁2：Agent.__init__ 兼容 (unique_id, model) 签名 ────────
        try:
            from mesa import Agent as MesaAgent

            if not hasattr(MesaAgent, '_patched_init'):
                _orig_init = MesaAgent.__init__

                def _compat_init(self, *args, **kwargs):
                    # Mesa 3.x: Agent.__init__(self, model)
                    # Mesa 2.x: Agent.__init__(self, unique_id, model)
                    if len(args) == 2 and not kwargs:
                        # (unique_id, model) 格式
                        unique_id, model = args
                        try:
                            _orig_init(self, model)
                        except TypeError:
                            _orig_init(self, unique_id, model)
                        self.unique_id = unique_id
                    elif len(args) == 1 and not kwargs:
                        # (model,) 格式
                        _orig_init(self, args[0])
                    else:
                        _orig_init(self, *args, **kwargs)

                MesaAgent.__init__ = _compat_init
                MesaAgent._patched_init = True
                print("[mesa_patch] ✅ Agent.__init__ 兼容补丁已应用")

        except ImportError:
            pass

    except Exception as e:
        print(f"[mesa_patch] ⚠️ 补丁应用失败（不影响运行）: {e}")


_patch_mesa()
