def conv_output_size(input_size, filter_size, padding=0, stride=1):
    # formula for output dimension:
    # O = (D -K +2P)/S + 1
    # where:
    #   D = input size (height/length)
    #   K = filter size
    #   P = padding
    #   S = stride
    return (input_size - filter_size + 2 * padding) // stride + 1
