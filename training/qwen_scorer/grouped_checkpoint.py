"""Group decoder checkpoints without changing module registration or attention semantics."""
import ast,inspect,textwrap,types,torch
from torch.utils.checkpoint import checkpoint

def enable_grouped_checkpoints(language_model,group_size=2,min_tokens=100000):
 if getattr(language_model,"_grouped_checkpoint",False):return
 original=inspect.unwrap(type(language_model).forward)
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
 fn.decorator_list=[d for d in fn.decorator_list if not (isinstance(d,ast.Name) and d.id=="auto_docstring")]
 loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=="decoder_layer" for x in n.target.elts))
 # RoPE uses three spatial/temporal axes, but FlashAttention packed-sequence
 # metadata must use the one-dimensional text sequence positions.
 for node in ast.walk(loop):
  if isinstance(node,ast.keyword) and node.arg=="position_ids":
   node.value=ast.Name(id="text_position_ids",ctx=ast.Load())
 # Keep the installed decoder call and attention mask selection.
 body=loop.body
 inner=ast.FunctionDef(name="_run_group",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="hidden_states"),ast.arg(arg="_layers")],kwonlyargs=[],kw_defaults=[],defaults=[ast.Name(id="_group",ctx=ast.Load())]),body=[ast.For(target=ast.Name(id="decoder_layer",ctx=ast.Store()),iter=ast.Name(id="_layers",ctx=ast.Load()),body=body,orelse=[]),ast.Return(value=ast.Name(id="hidden_states",ctx=ast.Load()))],decorator_list=[])
 replacement=ast.parse("""
for _start in range(0,self.config.num_hidden_layers,self._checkpoint_group_size):
 _group=tuple(self.layers[_start:_start+self._checkpoint_group_size])
 hidden_states=_group_checkpoint(_run_group,hidden_states,use_reentrant=True)
""").body[0]
 replacement.body.insert(1,inner)
 condition=ast.parse("self.training and torch.is_grad_enabled() and hidden_states.shape[1]>=self._checkpoint_min_tokens",mode="eval").body
 idx=fn.body.index(loop);fn.body[idx]=ast.If(test=condition,body=[replacement],orelse=[loop])
 namespace=dict(original.__globals__);namespace["_group_checkpoint"]=checkpoint
 ast.fix_missing_locations(tree)
 exec(compile(tree,"grouped_checkpoint_generated","exec"),namespace)
 language_model.forward=types.MethodType(namespace[fn.name],language_model)
 language_model._checkpoint_group_size=group_size
 language_model._checkpoint_min_tokens=min_tokens
 language_model._grouped_checkpoint=True
