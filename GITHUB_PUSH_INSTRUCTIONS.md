# GitHub Push Instructions

To push this repository to GitHub, you'll need to set up authentication. Here are two options:

## Option 1: SSH Authentication (Recommended)

1. Generate an SSH key (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. Add your SSH key to the ssh-agent:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. Add your SSH key to your GitHub account:
   - Copy your public key: `cat ~/.ssh/id_ed25519.pub`
   - Go to GitHub Settings > SSH and GPG keys > New SSH key
   - Paste your key and save

4. Change the remote URL to SSH:
   ```bash
   git remote set-url origin git@github.com:egargale/hmm_test.git
   ```

5. Push to GitHub:
   ```bash
   git push origin master
   ```

## Option 2: Personal Access Token

1. Generate a personal access token on GitHub:
   - Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
   - Generate new token with repo permissions
   - Copy the token

2. Push with token authentication:
   ```bash
   git push https://<token>@github.com/egargale/hmm_test.git master
   ```

Replace `<token>` with your actual personal access token.

## Verify the Push

After pushing, you can verify that your commit is on GitHub by visiting:
https://github.com/egargale/hmm_test