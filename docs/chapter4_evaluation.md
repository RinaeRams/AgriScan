# Chapter 4: Model Evaluation and Testing

## Training Results

| Metric | Value |
|--------|-------|
| Epochs run | 20 |
| Early stopping epoch (best val_loss) | 16 |
| Final train accuracy | 0.8873 (88.73%) |
| Final val accuracy | 0.9233 (92.33%) |
| Best val accuracy (epoch 16) | 0.9249 (92.49%) |
| Final train loss | 0.3236 |
| Final val loss | 0.2429 |

## Test Set Evaluation

| Metric | Value |
|--------|-------|
| Total test images | 2,971 |
| Overall accuracy | 92.12% |
| 95% CI | 91.10% - 93.04% |
| Top-3 accuracy | 99.23% |
| Macro F1 | 0.9111 |
| Macro Precision | 0.9061 |
| Macro Recall | 0.9197 |
| Weighted F1 | 0.9208 |
| Healthy vs Diseased accuracy | 99.02% |
| Consistency (5 runs) | 100.0% |

---

## (b) New Response-Time JSON

```json
{
  "endpoint": "/api/analyze-plant",
  "auth_method": "flask test client (no MongoDB auth)",
  "response_times_ms": {
    "mean": 558.4,
    "median": 545.6,
    "max": 692.5,
    "min": 445.0,
    "stdev": 73.7
  },
  "consistency": {
    "consistent": true,
    "unique_predictions": 1,
    "unique_disease": "Tomato_healthy"
  },
  "sample_results": [
    { "disease": "Tomato_healthy", "confidence": 96.51, "offline": false },
    { "disease": "Tomato_healthy", "confidence": 96.51, "offline": false },
    { "disease": "Tomato_healthy", "confidence": 96.51, "offline": false }
  ]
}
```

**Pre-warm optimization**: Model pre-warmed at Flask startup (dummy 224x224 prediction on boot). First-request latency eliminated. Steady-state median: 545.6ms (was ~7000ms+ on cold start before fix).

---

## (c) Render Health Result

| Property | Value |
|----------|-------|
| Service name | agriscan-app |
| Status | **Service Suspended** (Render free tier) |
| HTTP response | 503 Service Suspended |
| Health URL | https://agriscan-app.onrender.com/api/health |
| Local health check | PASS - ml_service: active, mongodb: connected |
| Auto-deploy on resume | Yes (GitHub commit 1b9846c) |
| Action required | Resume from [Render Dashboard](https://dashboard.render.com) |

---

## (d) Phone/Mobile Test Table

| Test Category | Test Case | Status | Details |
|--------------|-----------|--------|---------|
| **Responsive Layout** | CSS breakpoints | PASS | Media queries tested at 820px and 420px breakpoints |
| **Responsive Layout** | Login form | PASS | Form fields resize for mobile, single-column layout on narrow screens |
| **Responsive Layout** | Register button | PASS | Register button visible and clickable on mobile viewport |
| **API Endpoint** | `/api/analyze-plant` | PASS | Returns disease, confidence, all_predictions, recommendations, offline=false |
| **API Endpoint** | `/api/health` | PASS | Returns `status:healthy`, `ml_service:active`, `mongodb:connected` (local) |
| **API Endpoint** | `/login` POST | PASS | Returns `success:true`, `role:user` |
| **Consistency** | 10 repeated requests | PASS | 100% consistent (1 unique prediction: Tomato_healthy, 96.51% confidence) |
| **Full mobile device** | Physical device testing | Not Verified | Browser viewport testing via CSS inspection only; physical device testing requires devices |

---

## Preprocessing Comparison

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| [0, 1] (used during training) | 92.12% | 0.9111 |
| [-1, 1] | 76.54% | 0.7619 |

**Conclusion**: [0, 1] preprocessing is confirmed correct for this MobileNetV2 model.

---

## Per-Class Metrics Summary

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|  
| Pepper_bell_Bacterial_spot | 0.9603 | 0.9732 | 0.9667 | 149 |
| Pepper_bell_healthy | 0.9817 | 0.9683 | 0.9749 | 221 |
| Potato_Early_blight | 0.9667 | 0.9667 | 0.9667 | 150 |
| Potato_healthy | 0.8077 | 0.9545 | 0.8750 | 22 |
| Potato_Late_blight | 0.9375 | 0.9000 | 0.9184 | 150 |
| Tomato_Target_Spot | 0.9096 | 0.7667 | 0.8320 | 210 |
| Tomato_mosaic_virus | 0.8209 | 1.0000 | 0.9016 | 55 |
| Tomato_YellowLeaf_Curl_Virus | 0.9935 | 0.9605 | 0.9767 | 481 |
| Tomato_Bacterial_spot | 0.9226 | 0.9718 | 0.9466 | 319 |
| Tomato_Early_blight | 0.8029 | 0.7333 | 0.7666 | 150 |
| Tomato_healthy | 0.9492 | 0.9333 | 0.9412 | 120 |
| Tomato_Late_blight | 0.9314 | 0.9021 | 0.9165 | 286 |
| Tomato_Leaf_Mold | 0.8951 | 0.9014 | 0.8982 | 142 |
| Tomato_Septoria_leaf_spot | 0.8430 | 0.9321 | 0.8853 | 265 |
| Tomato_Spider_mites | 0.8699 | 0.9323 | 0.9000 | 251 |

## Most Confused Class Pairs

| True Class | Predicted Class | Count |
|------------|-----------------|-------|
| Tomato__Target_Spot | Tomato_Spider_mites | 19 |
| Tomato_Early_blight | Tomato_Septoria_leaf_spot | 14 |
| Tomato__Target_Spot | Tomato_Septoria_leaf_spot | 11 |
| Tomato_Late_blight | Tomato_Early_blight | 11 |
| Tomato_Early_blight | Tomato_Bacterial_spot | 8 |

---

## Status Changes (Chapter 4)

- **Partly Met** rows updated to **Fully Met** where evidence confirms
- **Not Verified** rows updated to **PASS** for local testing or **Not Verified** for requiring physical devices
- All API response time measurements now reflect the pre-warm optimization
- Healthy vs diseased binary classification achieves 99.02% accuracy (strong diagnostic capability)
