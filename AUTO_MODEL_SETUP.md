# Additional Three-Wheeler Detection Model

The application uses the standard YOLOv8 detector for car, motorcycle, bus and truck detection. It can additionally load `best.pt` for South-Asian three-wheeler / auto-rickshaw detection.

Place the optional `best.pt` file in the project root.

The dedicated model's class `2` (3-wheeler) is displayed and counted as `auto`. When detections from both models overlap, the dedicated auto detection takes precedence to reduce double-counting.

Run the application with:

```bash
streamlit run unified_app.py
```

The optional fine-tuned model is not included in this copy because the available GitHub file interface cannot transfer binary model weights. Refer to the upstream repository for the file and its provenance.
