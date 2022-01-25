"""
    该版本没有valid过程
"""
import torch
import matplotlib.pyplot as plt


class Trainner():
    def __init__(self):
        """
            init trainner
        """
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
        
        torch.save(checkpoint, 'output/checkpoints/model_state.pt')
    

    def load_checkpoint(self, path, model, optimizer):
        # load check point 
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        min_val_loss = checkpoint['min_val_loss']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

        return model, optimizer, epoch, min_val_loss

    
    def training(self, model, device, epochs, 
                 id_train_dl, core_cust_id_train_dl, prod_code_train_dl, dense_train_dl,
                 id_test_dl, core_cust_id_test_dl, prod_code_test_dl, dense_test_dl,
                 criterion, optimizer):
        """
            zip all id dataloader
        """
        training_losses = []

        # 模型中有BN层(Batch Normalization)和Dropout,需要在训练时添加model.train()
        model.train()

        def plotting_loss(training_losses=None, validation_losses=None):
            """
                plotting train | validtation loss
                画图部分要改成每个iter看一次
            """
            if training_losses:
                epoch_count = range(1, len(training_losses)+1)
                plt.plot(epoch_count, training_losses, 'r--')
                plt.title(['Training Loss'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig('output/training_loss.jpg')
            if validation_losses:
                epoch_count = range(1, len(validation_losses)+1)
                plt.plot(epoch_count, validation_losses, 'b--')
                plt.title(['Validation Loss'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig('output/validation_loss.jpg')

        train_dl = zip(id_train_dl, core_cust_id_train_dl, prod_code_train_dl, dense_train_dl)
        # Training
        running_training_loss = 0.0
        for _ in range(epochs): 
            for id_batch, core_cust_id_batch, prod_code_batch, (dense_batch, y_batch) in train_dl:
                    # Convert to Tensors
                    """
                        embedding层的输入要是long或int型不能是float
                    """
                    id_batch = id_batch.long().to(device)
                    core_cust_id_batch = core_cust_id_batch.long().to(device)
                    prod_code_batch = prod_code_batch.long().to(device)
                    dense_batch = dense_batch.float().to(device)
                    y_batch = y_batch.float().to(device)

                    optimizer.zero_grad()
                    
                    # Make prediction
                    output = model(
                                    id_input = id_batch,
                                    core_cust_id_input = core_cust_id_batch,
                                    prod_code_input = prod_code_batch,
                                    dense_input = dense_batch
                                    )
                    # Calculate loss
                    loss = criterion(torch.squeeze(output), y_batch)
                    loss.backward()
                    running_training_loss += loss.item()
            # Average loss across timesteps
            training_losses.append(running_training_loss / len(dense_train_dl)) # 数据对齐长度一致只需要随便取一个dl即可

        # Visualize loss
        plotting_loss(training_losses)

        # Make test prediction
        output = model(
                        id_input = id_test_dl,
                        core_cust_id_input = core_cust_id_test_dl,
                        prod_code_input = prod_code_test_dl,
                        dense_input = dense_test_dl
                        )
        
        return output