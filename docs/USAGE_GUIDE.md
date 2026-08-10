# VLM Online DBiLD Runtime 1.2.0 使用指南

本版在 Runtime 1.1.0 的離線模型封裝上新增 transition endpoint。模型只在 FastAPI application startup 載入一次，parsing、text 與 transition 共用同一個 model、processor、generation 設定與 semaphore；transition 是單次 multi-image 推論，不會重新 deployment。

## 1. 版本與固定契約

| 項目 | 值 |
|---|---|
| Package | \`VLM_Online_DBiLD_runtime_1.2.0\` |
| Source commit | \`b7e918945cd5f019ec092f3aa518b2839845f22e\` |
| Student | \`Qwen3-VL-8B-Instruct\` |
| Student revision | \`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b\` |
| Adapter | \`stage1_a4_r32_attn_mlp_projector\` |
| Deployment | \`4bit_base_bf16_adapter\` |
| Compute | BF16 |
| Device map | \`auto\` |
| Max concurrency | 1，共用 semaphore |
| Generation | \`max_new_tokens=2048\`, \`do_sample=false\` |
| Offline | \`HF_HUB_OFFLINE=1\`, \`TRANSFORMERS_OFFLINE=1\`, \`HF_DATASETS_OFFLINE=1\` |
| Uvicorn | 1 worker |

已驗證環境為 4× RTX 4000 Ada，模型分布於 \`cuda:0\`–\`cuda:3\`。最低 GPU VRAM 需求為8.7GB。

## 2. 系統需求

需要 Linux、Docker、NVIDIA Driver、NVIDIA Container Toolkit、CUDA 12.4 相容環境、足夠 RAM/磁碟空間與可用 port。檢查：

~~~bash
nvidia-smi
docker version
docker info
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 nvidia-smi
df -h
~~~

Runtime 模型、processor、tokenizer、adapter 均已嵌入 package；Docker image 不含權重，啟動時從 package 唯讀掛載，因此 inference 不需要 Hugging Face 下載。

## 3. 解壓與 checksum

在 archive 所在目錄：

~~~bash
sha256sum VLM_Online_DBiLD_runtime_1.1.0_<sha>_<timestamp>.tar.gz
tar -xzf VLM_Online_DBiLD_runtime_1.1.0_<sha>_<timestamp>.tar.gz
cd VLM_Online_DBiLD_runtime_1.1.0_<sha>_<timestamp>
sha256sum -c checksums.sha256
(cd models/student && sha256sum -c checksums.sha256)
(cd models/adapter && sha256sum -c checksums.sha256)
~~~

若需人工檢查 adapter：

~~~bash
sha256sum models/adapter/adapter_model.safetensors
~~~

其值必須符合 release manifest。不要修改權重內容。

## 4. Package 結構

~~~text
app/                 推論服務與必要 runtime modules
config/runtime.yaml  既有正式 config；route 內部決定 mode
models/student/      離線 Student、processor/tokenizer 與權重
models/adapter/      adapter、deployment metadata 與 projector
samples/             sample image
scripts/             build、run、healthcheck、smoke test、stop
docs/API.md          API 契約
docs/DEPLOYMENT.md   部署需求
reports/             build 與 runtime 證據
Dockerfile           固定 CUDA/Python image 定義
~~~

## 5. Build

~~~bash
bash scripts/build.sh
~~~

預設 image tag 為 \`vlm-online-dbild-runtime:1.1.0\`。可用 \`VLM_IMAGE_NAME\` 覆寫：

~~~bash
VLM_IMAGE_NAME=registry.example/vlm-runtime:1.1.0 bash scripts/build.sh
docker image inspect vlm-online-dbild-runtime:1.1.0
~~~

Build context 僅包含 app、config、scripts 與依賴；Dockerfile 不會 \`COPY models\`。Build log 寫入 \`reports/build.txt\`。image 內含 build-essential、bitsandbytes/Triton 所需 compiler 與鎖定 Python 依賴。

## 6. 啟動

~~~bash
bash scripts/run.sh
~~~

實際掛載：

~~~text
models/student/      -> /models/student:ro
models/adapter/      -> /models/adapter:ro
config/runtime.yaml  -> /config/runtime.yaml:ro
samples/             -> /data:ro
output/              -> /output
~~~

\`run.sh\` 使用 \`--gpus all\`、\`--network none\`、一個 Uvicorn worker。預設值：

| 變數 | 預設值 | 用途 |
|---|---|---|
| \`VLM_IMAGE_NAME\` | \`vlm-online-dbild-runtime:1.1.0\` | image tag |
| \`VLM_CONTAINER_NAME\` | \`vlm-online-dbild-runtime-110\` | container 名稱，health/smoke/stop 要一致 |
| \`VLM_PORT\` | \`8000\` | host published port |

目前腳本固定 \`--gpus all\`，沒有 GPU 選擇參數。不要啟動第二個 Uvicorn worker 或第二個 runtime container。

## 7. Startup hard checks 與共享模型

entrypoint 先檢查 Student/adapter 檔案、adapter hash、projector \`modules_to_save\`、4-bit artifact mode、compiler、CUDA、BF16、bitsandbytes、Triton、cache 與 offline variables。模型載入發生在 FastAPI lifespan startup，endpoint 內不執行 \`from_pretrained\`、\`PeftModel.from_pretrained\` 或 adapter load。

啟動後 log 會包含：

~~~text
model_instance_id=<opaque id>
processor_instance_id=<opaque id>
model_load_count=1
supported_output_modes=parsing,text
~~~

每個 endpoint request 也會記錄相同的 \`model_instance_id\`。/ready 會回傳 \`model_load_count=1\`、同一 instance id 及兩個 endpoint。兩種模式共用同一個 \`asyncio.Semaphore(1)\`；同時只有一筆 GPU inference。

## 8. Health 與 Ready

~~~bash
bash scripts/healthcheck.sh
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
~~~

\`/health\` 只表示 process 存活。預期 \`{"status":"ok"}\`。\`/ready\` 必須在 Student、processor、adapter、projector、GPU runtime 均完成後才成功，結構包含：

~~~json
{
  "status": "ready",
  "model_loaded": true,
  "model_load_count": 1,
  "model_instance_id": "opaque-runtime-id",
  "adapter_loaded": true,
  "projector_restored": true,
  "supported_output_modes": ["parsing", "text"],
  "endpoints": {
    "text": "/infer/text",
    "parsing": "/infer/parsing"
  },
  "merged_artifact_mode": "4bit_base_bf16_adapter",
  "device_map": {}
}
~~~

## 9. Parsing endpoint

~~~bash
curl -X POST http://127.0.0.1:8000/infer/parsing \
  -F 'image=@samples/sample_ui_0001.png;type=image/png' \
  -F 'instruction=List all visible interactive UI elements on this screen.'
~~~

輸入是 multipart 的 \`image\`、\`instruction\`、可選 \`query\` 與 \`request_id\`。若同時提供 instruction/query，instruction 優先。服務使用既有 \`compose_prompt(..., output_mode="parsing")\`，再套用 parsing output processor。

回應包含 \`raw_output\`、\`usable\`、\`parse_error\`、\`coordinate_system\`、\`elements\`、\`inference_debug\`、\`id\`、\`query\`、\`elapsed_seconds\`。核心 schema：

~~~json
{
  "coordinate_system": "normalized_0_1000",
  "elements": [
    {
      "text": "Settings",
      "bbox_norm": [100, 200, 300, 280],
      "focused": true
    }
  ]
}
~~~

\`bbox_norm\` 每個數值為 0–1000意思是1k畫面下x[0,1920],y[0,1080]都映射到[0,1000]，\`elements\` 是 list，\`focused\` 是 boolean。實際元素由圖片與模型輸出決定。

## 10. Text endpoint

~~~bash
curl -X POST http://127.0.0.1:8000/infer/text \
  -F 'image=@samples/sample_ui_0001.png;type=image/png' \
  -F 'instruction=Describe the current screen.'
~~~

服務使用既有 \`compose_prompt(..., output_mode="text")\`，呼叫同一 processor/model/semaphore，但不套用 parsing JSON processor。成功回應的 \`text\` 與 \`raw_output\` 都是非空自然語言文字，例如：

~~~json
{
  "id": "request-id",
  "query": "Describe the current screen.",
  "text": "The screen shows ...",
  "raw_output": "The screen shows ...",
  "usable": true,
  "elapsed_seconds": 12.34,
  "inference_debug": {
    "mode": "text",
    "model_instance_id": "opaque-runtime-id",
    "generation_kwargs": {
      "do_sample": false,
      "max_new_tokens": 2048
    }
  }
}
~~~

Text response 不要求 \`coordinate_system\`、\`elements\`、\`bbox_norm\` 或 \`focused\`。

## 11. Legacy 與 forbidden fields

\`POST /infer\` 是 deprecated 的 parsing alias，等同 \`POST /infer/parsing\`，不會由 request 動態切換 mode。

以下欄位對兩個 endpoint 都會回 HTTP 422：

~~~text
prompt
prompt_template
system_prompt
output_mode
max_new_tokens
do_sample
temperature
top_p
generation_config
~~~

例如：

~~~bash
curl -i -X POST http://127.0.0.1:8000/infer/text \
  -F 'image=@samples/sample_ui_0001.png;type=image/png' \
  -F 'instruction=Describe the screen.' \
  -F 'output_mode=parsing'
~~~

generation 固定為 \`max_new_tokens=2048\`、\`do_sample=false\`；caller 不能覆寫完整 prompt 或 generation 參數。

## 12. 驗證與 logs

完整 smoke test：

~~~bash
bash scripts/smoke_test.sh
~~~

它依序驗證 /ready、parsing、text、legacy alias、共用 model instance、parsing schema、text 非空及兩個 endpoint 的 forbidden fields 422。報告輸出到 \`reports/health.json\`、\`reports/inference_parsing.json\`、\`reports/inference_text.json\`、\`reports/inference_legacy.json\`、\`reports/forbidden_fields.txt\` 與 \`reports/schema_validation.txt\`。

常用診斷：

~~~bash
docker ps -a
docker logs vlm-online-dbild-runtime-110
docker logs -f vlm-online-dbild-runtime-110
docker exec vlm-online-dbild-runtime-110 nvidia-smi
docker inspect vlm-online-dbild-runtime-110
~~~

/ready 失敗時先看模型載入 traceback、GPU VRAM、adapter 權限、Triton cache、compiler 與 offline variables。CUDA OOM 時確認 GPU process、\`device_map\` 及沒有啟動第二個 request worker。

## 13. 停止與更新

~~~bash
bash scripts/stop.sh
docker ps -a
~~~

\`stop.sh\` 會停止並移除預設 container。重啟執行 \`run.sh\` 即可。更換 Student revision、adapter、source commit 或 runtime contract 時，應建立新 Runtime 版本；不可覆寫 1.0.0。

## 14. Transition endpoint

`POST /infer/transition` 使用 `before_image` 與 `after_image`；前者永遠是 Image 1，後者永遠是 Image 2。兩張圖都必須是 PNG/JPEG、各不超過 10 MiB 與 20,000,000 decoded pixels，服務會做 EXIF transpose 並轉 RGB。`instruction` 優先於 `query`；caller 不可傳 `output_mode` 或完整 prompt。

Transition 會在同一個 user message 內按 before、after、text 順序送入 processor，執行單次 multi-image processor call 與單次 `model.generate`。它和 parsing/text 共用一份模型、processor、單一 inference queue/semaphore 與 GPU 配置，不會重新 deployment；雙圖可能比單圖需要更多 VRAM。回應的 `transition` 只接受嚴格驗證的 schema，解析失敗時回 `usable=false`、`parse_error` 與 `raw_output`。
