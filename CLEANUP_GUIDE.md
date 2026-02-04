# Git Repository Cleanup Guide

Your repository is currently **2.97 GiB** because large files exist in git history even though they're no longer tracked.

## Current Status ✅
- Updated `.gitignore` to exclude dist/, snapshots/, binaries, and images
- Removed these files from tracking (they won't be added to future commits)
- Local files remain untouched

## To Truly Clean the Repository

### Option 1: Use BFG Repo-Cleaner (Easiest & Fastest) ⭐

1. **Download BFG**: https://rtyley.github.io/bfg-repo-cleaner/
   ```powershell
   # Or install via chocolatey
   choco install bfg-repo-cleaner
   ```

2. **Run cleanup**:
   ```powershell
   # Clone a fresh bare repo
   git clone --mirror https://github.com/Manalokosdev/Ribossome.git ribossome-cleanup.git
   cd ribossome-cleanup.git
   
   # Remove files larger than 1MB from history
   bfg --strip-blobs-bigger-than 1M
   
   # Or delete specific folders from history
   bfg --delete-folders "{dist,snapshots}" --no-blob-protection
   
   # Clean up
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   
   # Force push (BREAKS EXISTING CLONES!)
   git push --force
   ```

3. **Update your local repo**:
   ```powershell
   cd C:\Filipe\ALsimulatorv3
   git fetch origin
   git reset --hard origin/main
   ```

**Expected result**: Repo should shrink to < 100 MB

---

### Option 2: Use git-filter-repo (More Control)

1. **Install**: `pip install git-filter-repo`

2. **Backup first!**:
   ```powershell
   cd ..
   Copy-Item -Recurse ALsimulatorv3 ALsimulatorv3-backup
   ```

3. **Clean specific paths**:
   ```powershell
   cd ALsimulatorv3
   git filter-repo --path dist/ --invert-paths --force
   git filter-repo --path snapshots/ --invert-paths --force
   git filter-repo --path-glob '*.exe' --invert-paths --force
   git filter-repo --path-glob '*.zip' --invert-paths --force
   ```

4. **Re-add remote and push**:
   ```powershell
   git remote add origin https://github.com/Manalokosdev/Ribossome.git
   git push --force --all
   git push --force --tags
   ```

---

### Option 3: Start Fresh (Nuclear Option)

If you don't care about history:

1. **Save current state**:
   ```powershell
   cd C:\Filipe\ALsimulatorv3
   Remove-Item -Recurse -Force .git
   ```

2. **Create new repo**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Ribossome v0.1.0"
   ```

3. **Force push to GitHub**:
   ```powershell
   git remote add origin https://github.com/Manalokosdev/Ribossome.git
   git push -u origin main --force
   ```

**Pros**: Cleanest slate, minimal repo size
**Cons**: Loses all commit history

---

## What NOT to Do ❌

- Don't use `git rm` on old commits (doesn't remove from history)
- Don't manually edit `.git/` folder (dangerous)
- Don't run cleanup without a backup

## After Cleanup

1. **Team notification**: Anyone with existing clones must re-clone
2. **Size check**: `git count-objects -vH` (should be < 100 MB)
3. **Verify**: All your code should still be there, just without bloat

## Prevention (Already Done ✅)

- `.gitignore` now excludes dist/, *.zip, *.exe, *.png
- Use `package_release.ps1` to generate distribution packages locally
- Snapshots are generated locally and not tracked
