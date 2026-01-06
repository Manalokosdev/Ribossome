# Branch Protection Setup

The repository now has two branches:
- **`main`** - Stable releases and public distribution
- **`develop`** - Your ongoing development work (current branch)

## Quick Setup (2 minutes):

### Protect the `main` branch:

1. **Go to:** https://github.com/Manalokosdev/Ribossome/settings/branch_protection_rules/new

2. **Branch name pattern:** Type `main`

3. **Enable only these 2 options:**
   - ☑ **Require a pull request before merging**
     - Set "Required approvals" to: **0**
   - ☑ **Do not allow bypassing the above settings**
     (This is at the bottom - makes rules apply to you too)

4. **Click "Create"** at the bottom

**That's it!** Everything else stays unchecked.

### Leave `develop` unprotected:
No setup needed - it's your working branch where you can push freely.

---

## Your Workflow:

### Daily work (you're already on develop):
```bash
git add .
git commit -m "your changes"
git push
```

### When ready to release to main:
```bash
git checkout main
git merge develop
git push
git checkout develop
```

Done! Simple workflow, protected stable branch.
