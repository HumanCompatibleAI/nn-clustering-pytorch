import torch

# Custom Data Loader to randomly generate data for simple networks


class SimpleDataLoader:
    def __init__(self,
                 fns,
                 input_type="single",
                 lim=5,
                 batch_size=250,
                 num_batches=500):
        """
        fns ([Tensor(Float)->Tensor(Float)]): List of functions to be
            approximated. Each acts element wise on a tensor
        input_type (str): One of "single", "multi". Single means a single input
            x and output f1(x), f2(x),..., multi means a different input for
            each function
        lim (Float): Inputs will be output uniformly between -lim and lim

        Each batch is generated iid at random, and this terminates after
        num_batches batches are requested
        """
        self.fns = fns
        self.input_type = input_type
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.out = len(fns)
        self.lim = lim
        self.count = 0

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        self.reset()
        return self

    def reset(self):
        self.count = 0

    def __next__(self):
        if self.count >= self.num_batches:
            raise StopIteration
        if self.input_type == "single":
            x = (torch.rand(self.batch_size) - 0.5) * 2 * self.lim
            y = torch.stack([fn(x) for fn in self.fns], axis=1)
        elif self.input_type == "multi":
            x = (torch.rand(self.batch_size, self.out) - 0.5) * 2 * self.lim
            y = torch.stack([self.fns[i](x[:, i]) for i in range(self.out)],
                            axis=1)
        self.count += 1
        return (x, y)
