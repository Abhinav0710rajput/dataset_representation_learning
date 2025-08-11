### **Training**

#### **Frobenius Norm**

  * **Finetune**: Use pre-trained dataset2vec model.

    ```bash
    python train_fr.py --split 1 2 3 4 5 --mode ft
    ```

  * **Direct Train**: Train from scratch.

    ```bash
    python train_fr.py --split 1 2 3 4 5 --mode dt
    ```

-----

#### **Result Space Prediction**

  * **Finetune**: Use pre-trained dataset2vec model.

    ```bash
    python train_rs.py --split 1 2 3 4 5 --mode ft
    ```

  * **Direct Train**: Train from scratch.

    ```bash
    python train_rs.py --split 1 2 3 4 5 --mode dt
    ```
