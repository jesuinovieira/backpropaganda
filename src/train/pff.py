import torch
import torch.nn.functional as F


def pff(
    model,
    train_loader,
    optimizer_rep,
    optimizer_gen,
    device,
    n_epochs=10,
    timesteps=10,
    hidden_dim=120,
    one_hot=lambda x: F.one_hot(x, num_classes=10).float(),
):
    print("Training...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pos = one_hot(y).to(device)

            # Negative (wrong) labels
            y_neg_idx = torch.randint(0, 10, y.shape, device=device)
            y_neg_idx[y_neg_idx == y] = (y_neg_idx[y_neg_idx == y] + 1) % 10
            y_neg = one_hot(y_neg_idx).to(device)

            # Positive phase
            h_pos = torch.zeros(x.size(0), hidden_dim, device=device)
            for _ in range(timesteps):
                h_pos, recon_pos = model(x, y_pos, h_pos)
            g_pos = model.compute_goodness(h_pos)

            # Negative phase
            h_neg = torch.zeros(x.size(0), hidden_dim, device=device)
            for _ in range(timesteps):
                h_neg, recon_neg = model(x, y_neg, h_neg)
            g_neg = model.compute_goodness(h_neg)

            # Losses
            loss_rep = F.softplus(g_neg - g_pos).mean()
            recon_target = torch.cat([model.extract_features(x).detach(), y_pos], dim=1)
            loss_gen = F.mse_loss(recon_pos, recon_target)

            total_batch_loss = loss_rep + loss_gen
            optimizer_rep.zero_grad()
            optimizer_gen.zero_grad()
            total_batch_loss.backward()
            optimizer_rep.step()
            optimizer_gen.step()

            total_loss += total_batch_loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss / len(train_loader):.4f}")
