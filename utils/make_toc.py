import re
import os
import argparse
import json
from pathlib import Path

TOC_LIST_PREFIX = "-"

def create_index(cwd, path_parameter):
    """ create markdown index of all markdown files in cwd and sub folders
    path_parameter: Path to the json file containing the metadata of the
    project
    """
    with open(path_parameter) as json_file:
        parameters = json.load(json_file)

    github_link = os.path.join("https://github.com/", parameters['GLOBAL']['GITHUB']['owner'],
                               parameters['GLOBAL']['GITHUB']['repo_name']
                               )

    base_len = len(cwd)
    base_level = cwd.count(os.sep)
    md_lines = []
    md_exts = ['.markdown', '.mdown', '.mkdn', '.mkd', '.md', '.html', '.py']
    for root, dirs, files in os.walk(cwd):
        if os.path.basename(root) not in ['creds', 'documentation']:

            files = sorted([f for f in files if not f[0] == '.' and os.path.splitext(f)[-1] in md_exts])
            dirs[:] = sorted([d for d in dirs if not d[0] == '.'])
            #try:
            #    files.remove('README.md')
            #except:
            #    pass
            ### find where the repo begins
            base_len = re.search(
            r'{}'.format(parameters['GLOBAL']['GITHUB']['repo_name']), root
            ).start() + len(parameters['GLOBAL']['GITHUB']['repo_name'])
            if len(files) > 0:
                level = root.count(os.sep) - base_level
                indent = '  ' * level
                if root != cwd:
                    indent = '  ' * (level - 1)
                    md_lines.append('{0} {2} **{1}/**\n'.format(indent,
                                                                os.path.basename(root),
                                                                TOC_LIST_PREFIX))
                rel_dir = '.{1}{0}'.format(os.sep, root[base_len:])
                for md_filename in files:
                    indent = '  ' * level

                    if os.path.splitext(md_filename)[1] == '.html':
                        bitbucket = "https://htmlpreview.github.io/?"
                        md_lines.append('{0} {3} [{1}]({5}{4}/blob/master{2}{1})\n'.format(indent,
                                                                             md_filename,
                                                                             rel_dir[1:], ### remove dot
                                                                             TOC_LIST_PREFIX,
                                                                             github_link,
                                                                             bitbucket))
                    else:
                        to_report = 'tree'
                        md_lines.append('{0} {3} [{1}]({4}/tree/master{2}{1})\n'.format(indent,
                                                                             md_filename,
                                                                             rel_dir[1:], ### remove dot
                                                                             TOC_LIST_PREFIX,
                                                                             github_link))
    return md_lines


def replace_index(filename, new_index, Header = "Project", add_description = False, path_parameter = None):
    """ finds the old index in filename and replaces it with the lines in new_index
    if no existing index places new index at end of file
    if file doesn't exist creates it and adds new index
    """

    pre_index = []
    post_index = []
    pre = True
    post = False
    try:
        with open(filename, 'r') as md_in:
            for line in md_in:
                if '<!-- filetree' in line:
                    pre = False
                if '<!-- filetreestop' in line:
                    post = True
                if pre:
                    pre_index.append(line)
                if post:
                    post_index.append(line)
    except FileNotFoundError:
        pass

    with open(filename, 'w') as md_out:
        top = README = """
# {}

""".format(Header.replace('_', ' '))

        if add_description:
            with open(path_parameter) as json_file:
                parameters = json.load(json_file)

            top += "\n\n{}".format(parameters['GLOBAL']['DESCRIPTION'])

        top += """

## Table of Content

"""


        md_out.writelines(top)
        md_out.writelines(pre_index)
        md_out.writelines(new_index)
        md_out.writelines(post_index[1:])
