class Pipeline:
    def __init__(self, model, optimizer, loss_function, train_loader, val_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def run(self, nr_epochs):
        train_losses = []
        val_losses = []

        for epoch in range(nr_epochs):
            train_loss = self.train()
            val_loss = self.val()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{nr_epochs}, Training Loss: {train_loss}\n')
            print(f'Epoch {epoch + 1}/{nr_epochs}, Validation Loss: {val_loss}\n')

        return train_losses, val_losses

    def train(self):
        self.model.train()
        total_loss = 0

        for start_image, out_image, time_skip in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(start_image, time_skip)
            loss = self.loss_function(outputs, out_image)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader.dataset)

    def val(self):
        total_loss = 0
        for start_image, out_image, time_skip in self.val_loader:
            outputs = self.model(start_image, time_skip)
            total_loss += self.loss_function(outputs, out_image).item()

        return total_loss / len(self.val_loader.dataset)
