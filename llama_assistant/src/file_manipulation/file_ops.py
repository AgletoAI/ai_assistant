
import os
import git

def read_file(path):
    with open(path, 'r') as file:
        return file.read()

def write_file(path, content):
    with open(path, 'w') as file:
        file.write(content)

def list_directory(path):
    return os.listdir(path)

def init_git_repo(path):
    repo = git.Repo.init(path)
    return repo

def commit_changes(repo, message):
    repo.git.add(A=True)
    repo.index.commit(message)
