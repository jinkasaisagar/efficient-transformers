import onnx, onnxruntime
from onnx import helper, TensorProto, shape_inference
import os
from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
import numpy as np
MODEL_ID = "google/diffusiongemma-26B-A4B-it"


def get_all_types(model):
    type_dict = {}
    for vi in model.graph.value_info:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.output:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for vi in model.graph.input:
        type_dict[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_dict[init.name] = init.data_type
    return type_dict

def expose_all_intermediate_outputs_robust(onnx_path, save_path=None):
    model = onnx.load(onnx_path, load_external_data=False)
    model = shape_inference.infer_shapes(model)

    existing_output_names = {o.name for o in model.graph.output}
    type_dict = get_all_types(model)

    added_outputs = []

    for node in model.graph.node:
        for out_name in node.output:
            if out_name in existing_output_names:
                continue
            dtype = type_dict.get(out_name, TensorProto.FLOAT)  # Default fallback
            vi = helper.make_tensor_value_info(out_name, dtype, shape=None)
            model.graph.output.append(vi)
            added_outputs.append(out_name)

    print(f"Added {len(added_outputs)} outputs to model.")
    if not save_path:
        save_path = str(onnx_path).replace(".onnx", "_debug.onnx")
    onnx.save(model, save_path)
    print(f"Saved debug model to {save_path}")
    return save_path

def analyze_tensor(name, tensor):
   FP16_MAX = 65504.0
   issues = []
   if not np.issubdtype(tensor.dtype, np.floating):
       return issues
   if np.isnan(tensor).any(): issues.append("NaNs")
   if np.isinf(tensor).any(): issues.append("Infs")
   if np.abs(tensor).max() >= FP16_MAX:
       issues.append(f"Overflow > {FP16_MAX}")
   return issues

def analyze_onnx_fp16_overflow(onnx_path, inputs: dict, json_output=None, stop_on_first=False):
    node_path = "fp32_nodes_" + (str(onnx_path).split("/")[-1].replace(".onnx", ".yaml"))
    onnx_path = str(onnx_path).replace(".onnx", "_debug.onnx")
    model = onnx.load(onnx_path, load_external_data=False)

    # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
    added_initializers = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_path))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                    np.array(0, np_tensor.dtype)
                )

    session_options = onnxruntime.SessionOptions()
    for name, value in added_initializers.items():
        session_options.add_initializer(name, value)

    # Get fetchable outputs
    valid_outputs = set()
    for vi in list(model.graph.value_info) + list(model.graph.output):
        valid_outputs.add(vi.name)

    output_names = []
    input_nodes = {}
    for node in model.graph.node:
        for out in node.output:
            if out in valid_outputs:
                output_names.append(out)

    # print(f'OUTPUT NAMES -> {output_names}')
    sess = onnxruntime.InferenceSession(onnx_path, session_options)

    session_input_names = [x.name for x in sess.get_inputs()]
    session_inputs = {}
    for inp_name in session_input_names:
        if inp_name in inputs.keys():
            session_inputs[inp_name] = inputs[inp_name]
    outputs = sess.run(output_names, session_inputs)
    name_to_tensor = dict(zip(output_names, outputs))

    print("\n--- FP16 Overflow Summary ---")
    problem_nodes = []
    for name in output_names:
        tensor = name_to_tensor[name]
        issues = analyze_tensor(name, tensor)
        if issues:
            print(f"[FP16 ISSUE] {name} | shape={tensor.shape} | max={np.abs(tensor).max():.2f} | {issues}")
            problem_nodes.append((name, issues))
            if stop_on_first:
                break  # early exit

    with open(node_path, "w") as f:
        f.write("FP32NodeInstanceNames:\n")
        for node_name in problem_nodes:
            f.write(f" - {node_name[0]}\n")

        nodes_with_flagged_inputs = 0
        # nodes that take flagged nodes as input should also be set to fp32
        for node in model.graph.node:
            for inp in node.input:
                if inp in [name for name, _ in problem_nodes]:
                    for out in node.output:
                        if not (out in [name for name, _ in problem_nodes]):
                            nodes_with_flagged_inputs += 1
                            print(f"Problematic Node {inp} is input to {out}")
                            f.write(f" - {out}\n")
            

    print(f"\nTotal FP16-unsafe outputs: {len(problem_nodes) + nodes_with_flagged_inputs}")
    return problem_nodes 

TEXT_PROMPT = "Why is the sky blue?"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer
chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

messages = [
    {"role": "user", "content": TEXT_PROMPT}
    ]

inputs = processor.apply_chat_template(
    messages,
    chat_template=chat_template,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding="max_length",
    max_length=256,
    truncation=True
)



# expose_all_intermediate_outputs_robust('/home/jsaisaga/qeff_llama/d_g_npi_agent_2layers/onnx-998a1cb621b818cb/DiffusionGemmaForBlockDiffusion.onnx')
# analyze_onnx_fp16_overflow('/home/jsaisaga/qeff_llama/d_g_npi_agent_2layers/onnx-998a1cb621b818cb/DiffusionGemmaForBlockDiffusion.onnx', inputs)
 