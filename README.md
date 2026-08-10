# VLM Online DBiLD Runtime 1.2.0

本封裝將同一個常駐的 Qwen3-VL-8B 4-bit + BF16 + adapter/projector model instance 暴露為三個固定用途 endpoint：

- `POST /infer/parsing`：結構化 UI parsing。
- `POST /infer/text`：自然語言文字回應。
- `POST /infer/transition`：同一 UI flow 的 before/after 雙圖狀態轉換。
- `POST /infer`：deprecated parsing alias。

快速開始：

```bash
bash scripts/build.sh
bash scripts/run.sh
bash scripts/healthcheck.sh
bash scripts/smoke_test.sh
bash scripts/stop.sh
```

模型只在 application startup 載入一次；三個 endpoint 共用同一 model instance、processor、GPU 配置與單一 inference queue (`asyncio.Semaphore(1)`)。Runtime 保持 `4bit_base_bf16_adapter`、`device_map=auto`、單併發、2048 tokens、BF16 與 Hugging Face offline mode。

Transition 的 `before_image` 永遠是 Image 1，`after_image` 永遠是 Image 2。它以同一 user message 進行單次 multi-image processor call 與單次 generate，不會重新 deployment；雙圖可能比單圖需要更多 VRAM。Caller 不能傳 `output_mode` 或完整 prompt。

完整 API、部署與驗證流程請見 [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) 與 [docs/API.md](docs/API.md)。

Source commit: `b7e918945cd5f019ec092f3aa518b2839845f22e`。
