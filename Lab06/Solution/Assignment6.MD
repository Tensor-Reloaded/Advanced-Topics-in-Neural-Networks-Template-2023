**Expected points: 4**

I only managed to do the first part of the homework, implementing the conv2d layer by hand. (tema.ipynb file)
The custom layer Handmade_conv2d_implementation accepts a single parameter: a weights tensor with the shape 
(2, 3, 4, 5) - (output_channels, input_channels, kernel_height, kernel_width)

If we apply this layer to an image of shape (1, 3, 10, 12) - (batch_size, input_channels, input_height, input_width)
we obtain an output tensor of shape (1, 2, 7, 8) - (batch_size, output_channels, output_height, output_width)

If we pass the input image through the torch.nn.functional.conv2d layer 
and through the Handmade_conv2d_implementation, the results we obtain 
are equal with a tolerance of 1e-5

I have also tried to implement a CNN for the CIFAR-10 dataset but did
not manage to get the validation accuracy high enough.