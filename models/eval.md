FROM EVALUATION
Classification Report:
              precision    recall  f1-score   support

          CN       1.00      0.96      0.98      6755
         MCI       0.91      0.99      0.95      1365
          AD       0.78      1.00      0.88       523

    accuracy                           0.97      8643
   macro avg       0.90      0.98      0.93      8643
weighted avg       0.97      0.97      0.97      8643

AUC-ROC for CN: 0.998
AUC-ROC for MCI: 0.999
AUC-ROC for AD: 1.000













FROM python src/train.py

The 'train_dataloader' does not 
have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=13` in the 
`DataLoader` to improve performance.
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.9173898100852966     │
│         test_loss         │    0.35066941380500793    │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 271/271 0:00:19 • 0:00:00 14.18it/s  
Training completed. Best model saved at: /Users/wongxinping/Documents/SUTD Y3T6/Applied deep learning/Project/ARIA/ARIA/models/ad-classifier-epoch=02-val_acc=0.99-v1.ckpt


Trainable params: 11.2 M                                                                                                                  
Non-trainable params: 0                                                                                                                   
Total params: 11.2 M                                                                                                                      
Total estimated model params size (MB): 44                                                                                                
Modules in train mode: 72                                                                                                                 
Modules in eval mode: 0                                                                                                                   
Total FLOPs: 0 