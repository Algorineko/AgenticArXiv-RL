"""Single-A40 synthetic context memory profile; no task text is read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

MIB = 1024**2

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(16 * MIB):
            h.update(block)
    return h.hexdigest()

def allowed(base, peak, total, free):
    predicted = base + 1.5 * (peak - base) * 1.5 + 2048
    return free >= 12 * 1024 and predicted < 0.6 * total - 2048, predicted

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, choices=range(4), required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.physical_gpu):
        raise SystemExit('Expected exactly the selected physical GPU')
    result = {'status': 'initializing', 'model_manifest_sha256': digest(a.manifest),
              'source_sha256': digest(Path(__file__)), 'physical_gpu': a.physical_gpu,
              'memory_fraction': 0.6, 'steps': [], 'max_verified_context_tokens': 0}
    def save():
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    try:
        manifest = json.loads(a.manifest.read_text())
        if manifest.get('model_id') != 'Qwen/Qwen3-8B' or not manifest.get('files'):
            raise ValueError('Unexpected manifest')
        for name, meta in manifest['files'].items():
            if Path(name).name != name or (a.model_dir / name).stat().st_size != meta['bytes'] or digest(a.model_dir / name) != meta['sha256']:
                raise ValueError('Model manifest verification failed')
        sample = subprocess.check_output(['nvidia-smi','-i',str(a.physical_gpu), '--query-gpu=memory.free,utilization.gpu,name','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(sample[0]) < 40000 or int(sample[1]) >= 10 or 'A40' not in sample[2]:
            result['status'] = 'gpu_unavailable'; return
        import torch
        from transformers import AutoModelForCausalLM
        from torch.nn.attention import sdpa_kernel, SDPBackend
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.backends.cuda.is_flash_attention_available() or not torch.backends.cuda.flash_sdp_enabled():
            result['status'] = 'flash_unavailable'; return
        torch.cuda.set_per_process_memory_fraction(.6, 0)
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16, device_map={'':0}, low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        model.eval()
        total = torch.cuda.get_device_properties(0).total_memory / MIB
        base = torch.cuda.memory_reserved(0) / MIB
        result.update(gpu_name=torch.cuda.get_device_name(0), total_mib=round(total,1), base_reserved_mib=round(base,1), initial_free_mib=int(sample[0]))
        for length in (4096,8192,12288):
            free = torch.cuda.mem_get_info(0)[0] / MIB
            if free < 12 * 1024:
                result['status'] = 'insufficient_free_memory'; break
            if length == 12288:
                ok, predicted = allowed(base, result['steps'][-1]['peak_reserved_mib'], total, free)
                result['predicted_12288_mib'] = round(predicted, 1)
                if not ok:
                    result['status'] = 'prediction_guard'; break
            torch.cuda.reset_peak_memory_stats(0)
            started = time.monotonic()
            ids = torch.full((1,length), 100, dtype=torch.long, device='cuda:0')
            with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False, use_cache=True, pad_token_id=model.config.eos_token_id)
            torch.cuda.synchronize()
            result['steps'].append({'input_tokens':length, 'peak_reserved_mib':round(torch.cuda.max_memory_reserved(0)/MIB,1), 'peak_allocated_mib':round(torch.cuda.max_memory_allocated(0)/MIB,1), 'seconds':round(time.monotonic()-started,2)})
            result['max_verified_context_tokens'] = length
            del ids, output
            torch.cuda.empty_cache()
            save()
        else:
            result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'cuda_allocation_failure' if 'out of memory' in str(exc).lower() else 'error'
        result['error_type'] = type(exc).__name__
        result['error'] = str(exc)[:500]
    finally:
        save()

if __name__ == '__main__':
    main()
