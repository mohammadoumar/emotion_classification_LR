def batch_tensor(tensor, batch_size):
    return [tensor[i:i+batch_size] for i in range(0, tensor.size(0), batch_size)]