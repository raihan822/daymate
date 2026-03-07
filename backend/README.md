# Personal notes
> **Note:** in Python project folder, all internal file imports should be kept relative import. at-least a dot. even if not necessary


## Github note:
### Development Phase:
```bash
git add .
git commit -m "Working on payment logic"
git push origin develop
```
- No deployment happens. Safe.

### When ready for production:
```bash
git checkout main
git merge develop
git push origin main
```
- Now Render deploys.