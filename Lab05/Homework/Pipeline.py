import torch
import wandb
from tqdm import tqdm

class Pipeline:

    @staticmethod
    def accuracy(output, labels):
        fp_plus_fn = torch.logical_not(output == labels).sum().item()
        all_elements = len(output)
        return (all_elements - fp_plus_fn) / all_elements

    @staticmethod
    def train(model, train_loader, config):
        training_loss = 0.0
        model.train()

        all_outputs = []
        all_labels = []

        for data, labels in train_loader:
            data = data.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)
            output = model(data)
            loss = config.criterion(output, labels)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            output = output.softmax(dim=1).detach().cpu().squeeze()
            labels = labels.cpu().squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

            training_loss += loss.item()

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        return (round(Pipeline.accuracy(all_outputs, all_labels), 4), loss)
    
    @staticmethod
    def val(model, val_loader, config):
        validation_loss = 0.0
        model.eval()

        all_outputs = []
        all_labels = []

        batch = 0
        for data, labels in val_loader:
            data = data.to(config.device, non_blocking=True)

            with torch.no_grad():
                output = model(data)

            loss = config.criterion(output, labels)
            output = output.softmax(dim=1).cpu().squeeze()
            labels = labels.squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

            validation_loss += loss
            config.writer.add_scalar(f"{config.legend} Val Loss/Batch", loss.item(), batch)
            batch += 1

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        return (round(Pipeline.accuracy(all_outputs, all_labels), 4), validation_loss)

    @staticmethod
    def do_epoch(model, train_loader, val_loader, config):
        acc, loss = Pipeline.train(model, train_loader, config)
        acc_val, loss_val = Pipeline.val(model, val_loader, config)
        # torch.cuda.empty_cache()
        return acc, loss, acc_val, loss_val
    
    @staticmethod
    def run(model, train_loader, val_loader, config):
        tbar = tqdm(tuple(range(config.epochs)))

        for epoch in tbar:
            acc, loss, acc_val, loss_val = Pipeline.do_epoch(model, train_loader, val_loader, config)
            tbar.set_postfix_str(f"Acc: {acc}, Acc_val: {acc_val}")
            config.writer.add_scalar(f"{config.legend} Train Accuracy/Epoch", acc, epoch)
            config.writer.add_scalar(f"{config.legend} Train Loss/Epoch", loss, epoch)
            config.writer.add_scalar(f"{config.legend} Val Accuracy/Epoch", acc_val, epoch)
            config.writer.add_scalar(f"{config.legend} Val Loss/Epoch", loss_val, epoch)
            config.writer.add_scalar(f"{config.legend} Model/Norm", model.get_norm(), epoch)

            config.writer.add_scalar("Learning rate/epoch", model.optimizer.param_groups[0]["lr"])
            config.writer.add_scalar("Batch size", next(iter(train_loader))[0].shape[0],)
            config.writer.add_text("Optimizer", str(model.optimizer),)

            wandb.log({"Val Accuracy/Epoch": acc_val, "Val Loss/Epoch": loss_val})