import os
import time
import torch
import numpy as np
from utils.plotting_utils import calculate_F2_score
from torch.utils.tensorboard import SummaryWriter

TRAIN_TIME = time.strftime("%Y-%m-%d", time.localtime())
TENSORBOARD_PATH = f'output/loss/tensorboard/{TRAIN_TIME}/'
if not os.path.exists(TENSORBOARD_PATH):
    os.makedirs(TENSORBOARD_PATH)
writer = SummaryWriter(TENSORBOARD_PATH)


class Trainner():
    def __init__(self):

        pass


    def save_checkpoint(self, epoch, min_val_loss, model_state, opt_state):
        """
            pytorch实现断点训练参考：https://zhuanlan.zhihu.com/p/133250753
        """
        print(f"New minimum reached at epoch #{epoch+1}, saving model state...")
        checkpoint = {
            'epoch' : epoch+1,
            'min_val_loss' : min_val_loss,
            'model_state' : model_state,
            'opt_state' : opt_state
        }
        
        torch.save(checkpoint, f'output/checkpoints/{TRAIN_TIME}_model_state.pt')
    

    def load_checkpoint(self, path, model, optimizer):
        # load check point 
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

        return model, optimizer, epoch, min_val_loss

    
    def training(self, model, device, epochs, train_dl, valid_dl, criterion, optimizer, validate_every=2):
        """
            zip all id dataloader
        """
        # save model according to min_validation_loss
        min_validation_loss = np.inf

        # 模型中有BN层(Batch Normalization)和Dropout,需要在训练时添加model.train()
        model.train()

        for epoch in range(epochs):
            print("Model training.......")
            running_training_loss = 0.0
            running_training_f2_score = 0.0
            # Training
            for core_cust_id_batch, prod_code_batch, (dense_batch, y_batch) in train_dl:
                    # Convert to Tensors
                    """
                        embedding层的输入要是long或int型不能是float
                    """
                    core_cust_id_batch = core_cust_id_batch.long().to(device)
                    prod_code_batch = prod_code_batch.long().to(device)
                    dense_batch = dense_batch.float().to(device)
                    y_batch = y_batch.float().to(device)

                    optimizer.zero_grad()
                    
                    # Make prediction
                    output = model(
                                   core_cust_id_input = core_cust_id_batch,
                                   prod_code_input = prod_code_batch,
                                   dense_input = dense_batch
                                  )
                    # Calculate Training loss
                    loss = criterion(torch.squeeze(output), y_batch)
                    loss.backward()
                    running_training_loss += loss.item()
                    # Calculate Training f2_score
                    f2_score = calculate_F2_score(torch.squeeze(output), y_batch)
                    running_training_f2_score += f2_score
            writer.add_scalar('training loss', running_training_loss / len(train_dl), epoch)
            writer.add_scalar('training f2 score', running_training_f2_score / len(train_dl), epoch)

            if epoch % validate_every == 0:
                print("Its time to validation......")
                # Set to eval mode
                model.eval()
                running_validation_loss = 0.0
                running_validation_f2_score = 0.0

                for core_cust_id_batch, prod_code_batch, (dense_batch, y_batch) in valid_dl:
                        # Convert to Tensors
                        core_cust_id_batch = core_cust_id_batch.long().to(device)
                        prod_code_batch = prod_code_batch.long().to(device)
                        dense_batch = dense_batch.float().to(device)
                        y_batch = y_batch.float().to(device)

                        output = model(
                                        core_cust_id_input = core_cust_id_batch,
                                        prod_code_input = prod_code_batch,
                                        dense_input = dense_batch
                                      )
                        # Calculate validation loss
                        validation_loss = criterion(torch.squeeze(output), y_batch)
                        running_validation_loss += validation_loss.item()
                        # Calculate validation f2_score
                        f2_score = calculate_F2_score(torch.squeeze(output), y_batch)
                        running_validation_f2_score += f2_score
            # Visualization
            writer.add_scalar('validation loss', running_validation_loss / len(valid_dl), epoch)
            writer.add_scalar('validation f2 score', running_validation_f2_score / len(valid_dl), epoch)

            is_best = running_validation_loss / len(list(valid_dl)) < min_validation_loss

            if is_best:
                min_validation_loss = running_validation_loss / len(valid_dl)
                self.save_checkpoint(
                                     epoch+1, 
                                     min_validation_loss, 
                                     model.state_dict(),
                                     optimizer.state_dict() 
                                    )  

