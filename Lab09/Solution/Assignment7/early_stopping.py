

class EarlyStopping:
    def __init__(self, epochs_permission, loss_difference=0.00015):
        self.epochs_permission = epochs_permission
        self.loss_difference = loss_difference
        self.counter = 0
        self.min_validation_loss = 1000000

    def verify_stopping_criteria(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.loss_difference:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.epochs_permission:
                return True
        self.min_validation_loss = validation_loss
        return False
