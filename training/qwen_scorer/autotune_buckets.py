"""Opt-in bucketing of FLA autotuning metadata; GPU equivalence test required before use."""
import ast,importlib,inspect,textwrap,json
SPECS=[
 ("fla.modules.l2norm",("l2norm_fwd","l2norm_bwd"),("l2norm_fwd_kernel","l2norm_bwd_kernel")),
 ("fla.modules.fused_norm_gate",("layer_norm_gated_fwd","layer_norm_gated_bwd"),("layer_norm_gated_fwd_kernel","layer_norm_gated_bwd_kernel")),
]
def validate_kernel_metadata(source,kernel_name):
 tree=ast.parse(textwrap.dedent(source));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==kernel_name)
 if any(isinstance(n,ast.Name) and n.id=="NB" and isinstance(n.ctx,ast.Load) for node in fn.body for n in ast.walk(node)):
  raise RuntimeError("NB participates in kernel arithmetic; refusing metadata-only patch")
def bucket_function_source(source,function_name):
 tree=ast.parse(textwrap.dedent(source));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name);fn.decorator_list=[]
 changes=0
 for node in ast.walk(fn):
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="NB":
   if ast.unparse(node.value)!="triton.cdiv(T, 2048)":raise RuntimeError("Unexpected NB definition")
   node.value=ast.Call(func=ast.Attribute(value=ast.Name(id="triton",ctx=ast.Load()),attr="next_power_of_2",ctx=ast.Load()),args=[node.value],keywords=[]);changes+=1
 if changes!=1:raise RuntimeError("Expected exactly one metadata key")
 ast.fix_missing_locations(tree)
 return tree
def enable_bucketed_autotune():
 count=0
 for module_name,functions,kernels in SPECS:
  module=importlib.import_module(module_name)
  for name in kernels:
   obj=getattr(module,name)
   while not inspect.isfunction(obj):obj=obj.fn
   validate_kernel_metadata(inspect.getsource(obj),name)
  for name in functions:
   fn=inspect.unwrap(getattr(module,name))
   if getattr(fn,"_ikv_bucketed_autotune",False):continue
   if fn.__code__.co_freevars:raise RuntimeError("Unexpected closure in FLA function")
   tree=bucket_function_source(inspect.getsource(fn),name)
   namespace=dict(fn.__globals__);exec(compile(tree,"bucketed_autotune_generated","exec"),namespace)
   patched=namespace[name]
   if patched.__code__.co_freevars:raise RuntimeError("Unexpected generated closure")
   fn.__code__=patched.__code__;fn._ikv_bucketed_autotune=True;count+=1
 print(json.dumps(dict(event="bucketed_autotune_enabled",functions=count)),flush=True)
 return count
