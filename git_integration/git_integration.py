import os
import shutil

from git import Repo

git_url = "https://github.com/OmerLiberman/resnet.git"
repo_dir = "/Users/omerliberman/Desktop/dir2"
branch = "master"

repo = None

def clone_from_git(git_url, repo_dir):
	"""
	clone from git to repo_dir.
	Make sure repo_dir is either (1) not exists directory,
	or (2) empty directory.
	"""
	global repo
	if os.path.isdir(repo_dir) and (len(os.listdir(repo_dir)) != 0):
		shutil.rmtree(repo_dir)
	repo = Repo.clone_from(git_url, repo_dir)

# clone_from_git(git_url, repo_dir)

### ----------------------

# def repo_to_tar(git_url, local_dir, is_dir_exists, tar_name='repo.tar'):
# 	with open(os.path.join(local_dir, tar_name), 'wb') as fp:
# 		Repo.archive(fp)
#
# repo_to_tar(git_url, repo_dir, False)

### ----------------------

def push_to_repo(to_push='.', commit="commit"):
	os.system("cd {directory}".format(directory=repo_dir))
	os.system("git add {to_push}".format(to_push=to_push))
	os.system("git commit -m {commit}".format(commit=commit))
	os.system("git push")

push_to_repo()