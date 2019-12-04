import torch

from torchsummary import summary

from reward_net import RewardNet

if __name__ == "__main__":
    # training set
    train_trajectories = [
        [[[1, 0, 1],
          [1, 0, 1],
          [1, 0, 1]],
         [[1, 0, 1],
          [1, 0, 1],
          [1, 0, 1]]
         ],

        [[[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]],
         [[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]
         ],

        [[[1, 0, 0],
          [1, 0, 1],
          [0, 0, 1]],
         [[1, 0, 0],
          [1, 0, 1],
          [0, 0, 1]]
         ],

        [[[0, 0, 0],
          [0, 1, 0],
          [0, 1, 0]],
         [[0, 0, 0],
          [0, 1, 0],
          [0, 1, 0]]
         ],

        [[[0, 0, 0],
          [1, 1, 0],
          [0, 0, 0]],
         [[0, 0, 0],
          [1, 1, 0],
          [0, 0, 0]]
         ],

        [[[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]],
         [[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]
         ],

        [[[1, 1, 1],
          [0, 0, 0],
          [1, 1, 1]],
         [[1, 1, 1],
          [0, 0, 0],
          [1, 1, 1]]
         ],
    ]

    # test set
    test_trajectories = [
        [[[1, 0, 0],
          [1, 0, 0],
          [1, 0, 0]],
         [[1, 0, 0],
          [1, 0, 0],
          [1, 0, 0]]
         ],

        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]],
         [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]
         ],

        [[[1, 1, 1],
          [0, 0, 0],
          [0, 0, 0]],
         [[1, 1, 1],
          [0, 0, 0],
          [0, 0, 0]]
         ],
    ]


    # use GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)    # for determinism, both on CPU and on GPU
    if torch.cuda.is_available():
        # required for determinism when using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ''' queste sono le traiettorie "hand_made", lo stato è un 3x3:
        la rete dà un'alto score se l'input presenta righe orizzontali
        invece lo score è basso se sono presenti righe verticali
        il test è fatto da esempi non presenti nel train '''
    # input_shape = 1, 3, 3
    # X_train = torch.Tensor(train_trajectories).reshape((len(train_trajectories), -1, *input_shape)).to(device)
    # X_test = torch.Tensor(test_trajectories).reshape((len(test_trajectories), -1, *input_shape)).to(device)

    ''' queste sono traiettorie random, lo stato è un 7x7. Il test è ottenuto aggiungendo del rumore ai primi 10 esempi del train '''
    input_shape = 1, 7, 7
    X_train = torch.randn((20, 8, 7, 7)).reshape(20, -1, *input_shape).to(device)
    X_test = X_train[:10] + torch.randn(X_train[:10].shape).to(device) / 10

    reward_net = RewardNet(input_shape).to(device)

    print("summary")
    print(summary(reward_net, input_shape, device=device))

    # evaluate before training
    reward_net.evaluate(X_test)

    # training
    reward_net.fit(X_train, max_epochs=300)

    # evaluate after training
    reward_net.evaluate(X_test)

    # save trained reward net
    torch.save(reward_net.state_dict(), "reward_net.pth") # TODO specificare file output da argomenti
