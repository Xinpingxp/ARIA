FROM EVALUATION
Classification Report:
              precision    recall  f1-score   support

          CN       0.99      0.91      0.95      6722
         MCI       0.71      0.94      0.81      1373
          AD       0.82      0.97      0.89       549

    accuracy                           0.92      8644
   macro avg       0.84      0.94      0.88      8644
weighted avg       0.93      0.92      0.92      8644

AUC-ROC for CN: 0.982
AUC-ROC for MCI: 0.980
AUC-ROC for AD: 0.998













FROM python src/train.py

The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=13` in the `DataLoader` to improve performance.

epoch 4: v_num: 7.000 train_loss: 0.458 val_loss: 0.172   
                                                                                         val_acc: 0.916
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