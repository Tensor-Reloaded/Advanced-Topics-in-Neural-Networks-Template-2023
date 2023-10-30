from tqdm import tqdm


def run(n, model, train_data, validation_data, test_data, loss_function, optimizer):  # n represents the number of
    # epochs
    train_loss_during_epochs = []
    validation_loss_during_epochs = []

    for epoch in range(n):
        for batch in train_data:
            train_input_tuples = []
            for index in range(len(batch[0])):
                train_item_1 = batch[0][index]
                train_item_2 = batch[1][index]
                train_item_3 = batch[2][index]
                train_input_element = (train_item_1, train_item_2, train_item_3)
                train_input_tuples.append(train_input_element)
            train_loss = train(model, train_input_tuples, loss_function, optimizer)

        for batch in validation_data:
            validation_input_tuples = []
            for index in range(len(batch[0])):
                validation_item_1 = batch[0][index]
                validation_item_2 = batch[1][index]
                validation_item_3 = batch[2][index]
                validation_input_element = (validation_item_1, validation_item_2, validation_item_3)
                validation_input_tuples.append(validation_input_element)
            validation_loss = val(model, validation_input_tuples, loss_function)

        train_loss_during_epochs.append(train_loss)
        validation_loss_during_epochs.append(validation_loss)

        print(f'Epoch {epoch + 1}/{n}, Training Loss: {train_loss}\n')
        print(f'Epoch {epoch + 1}/{n}, Validation Loss: {validation_loss}\n')

    for batch in test_data:
        test_input_tuples = []
        for index in range(len(batch[0])):
            test_item_1 = batch[0][index]
            test_item_2 = batch[1][index]
            test_item_3 = batch[2][index]
            test_input_element = (test_item_1, test_item_2, test_item_3)
            test_input_tuples.append(test_input_element)
        test_loss = val(model, test_input_tuples, loss_function)
    print(f'Test Loss: {test_loss}\n')

    return train_loss_during_epochs, validation_loss_during_epochs


def train(model, train_data, loss_function, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm(total=len(train_data), desc="Training", dynamic_ncols=True)

    for start_image, end_image, time_skip in train_data:
        optimizer.zero_grad()
        outputs = model((start_image, time_skip))
        loss = loss_function(outputs, end_image)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix({'Loss': loss.item()})
        pbar.update()

    pbar.close()
    return total_loss / len(train_data)


def val(model, validation_data, loss_function):
    total_loss = 0

    for start_image, end_image, time_skip in validation_data:
        outputs = model((start_image, time_skip))
        loss = loss_function(outputs, end_image)
        total_loss += loss.item()

    return total_loss / len(validation_data)


def test(model, test_data, loss_function):
    total_loss = 0

    for start_image, end_image, time_skip in test_data:
        outputs = model((start_image, time_skip))
        loss = loss_function(outputs, end_image)
        total_loss += loss.item()

    return total_loss / len(test_data)
