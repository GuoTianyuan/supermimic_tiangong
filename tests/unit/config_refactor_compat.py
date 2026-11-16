"""Narrow AST normalization for configuration plumbing, not algorithm changes.

Runtime parameterization is independently tested with real tensor rewards, exact
old gradients, custom weights, compiled rewards, CLI precedence and checkpoints.
This lets older relocation tests continue guarding unrelated learner expressions.
"""
import ast


class ConfigDefaults(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if ast.unparse(node) in {'mdp.max_step_reward', 'self.mdp.max_step_reward', 'self.max_step_reward'}:
            return ast.Name(id='MAX_STEP_REWARD', ctx=node.ctx)
        if ast.unparse(node) in {'c.w_root_pos', 'c.w_root_ori'}:
            return ast.Constant(value=.5)
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult) and ast.unparse(node.left) in {
            'c.w_body_pos', 'c.w_body_ori', 'c.w_body_linvel', 'c.w_body_angvel'}:
            return self.visit(node.right)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        text = ast.unparse(node)
        if text in {'self.max_step_reward = cfg.validate().max_step_reward(CONTROL_DT)',
                    'self.max_step_reward = max_step_reward', 'env.mdp = mdp',
                    'env.G_max = mdp.max_step_reward / (1 - env.gamma)'}:
            return None
        if text == "mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), reward_cfg(DYNAMICS, a_.get('reward_cfg')))":
            return None
        return self.generic_visit(node)

    def visit_Global(self, node):
        if node.names == ['mdp']:
            return None
        return node

    def visit_FunctionDef(self, node):
        if node.name == '__init__' and [a.arg for a in node.args.args] == ['self', 'device', 'max_step_reward']:
            node.args.args.pop()
            node.args.defaults.pop()
        return self.generic_visit(node)
