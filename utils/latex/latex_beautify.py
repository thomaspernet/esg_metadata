import re, os, tex2pix, json
import numpy as np
from PyPDF2 import PdfFileMerger
from wand.image import Image as WImage


def beautify(
table_number,
multi_lines_dep = None,
new_row = False,
multicolumn = None,
table_nte = None,
reorder_var = None,
jupyter_preview = True,
resolution = 150,
folder = 'Tables'):
    """
    Prepare a PDF table from multiple estimates

    Args:
    table_number: int
    multi_lines_dep: string. Add a new line under the dependant variables
    new_row: Boolean or list of string if not False: Add a new line below the column number
    Format is ["a", "b", "c"]. The first value will be used for the first column, which is the one on the left
    of the first column's estimate (1). So if there is 2 models estimate pass three value in the list.
    To avoid passing a value to the column on the left, use " & ", such as ['& test1', 'test2'].
    The first value will be used for the first column estimate (1 )
    multicolumn:  Dictionary: Group columns under the same label. New row added above the (1), (2)
    reorder_var:  Dictionary
    table_nte: string

    reorder_var -> a dic
    ## old position: new position. Should  be ordered. Start at 0
dic_ = {
    # Old, New
    8:4
}
Does not work for first two vars
    """
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    #table_number = 1
    table_in = "{}/table_{}.txt".format(folder,table_number)
    table_out = "{}/table_{}.tex".format(folder, table_number)

    regex_to_remove = \
    r"\s\sregimeELIGIBLE\s"

    with open('schema_table.json') as json_file:
        data = json.load(json_file)


    with open(table_in, "r") as f:
        lines = f.readlines()

        line_to_remove = []
    #return lines

    if table_nte!= None:
        max_ = 6
        max_1 = 9
    else:
        max_ =  8
        max_1 = 12

    for x, line in enumerate(lines[13:-max_]):
        test = bool(re.search(regex_to_remove, line))
        #test_1 = bool(re.search(comp, line))

        if test == True:
            line_to_remove.append(x + 13)
            line_to_remove.append((x + 13) + 1)


    with open(table_out, "w") as f:
        for x, line in enumerate(lines):

            if x not in line_to_remove:
                f.write(line)


    ### add ajdust box
    with open(table_out, 'r') as f:
        lines = f.readlines()

    if new_row != False:
        temp  = [' & '.join(new_row)]
        temp.append('\n \\\\[-1.8ex]')
        temp.append('\\\\\n ')
        #temp.append('\n \\\\[-1.8ex]\n')
        new_row_ = [temp[1] + temp[0] + temp[2] #+ temp[3]
        ]

    for x, line in enumerate(lines):
        label = bool(re.search(r"label",
                              line))
        tabluar = bool(re.search(r"end{tabular}",
                              line))
        if label:
            lines[x] = lines[x].strip() + '\n\\begin{adjustbox}{width=\\textwidth, totalheight=\\textheight-2\\baselineskip,keepaspectratio}\n'

        if tabluar:
            lines[x] = lines[x].strip() + '\n\\end{adjustbox}\n'

    if multi_lines_dep != None:

        for x, line in enumerate(lines):
            if x == 6:
                regex = r"(?<=\}}l).+?(?=\})" ### count number of c
                matches = re.search(regex, lines[x])

                nb_col = len(matches.group())
            if x == 9:
                to_add = "\n&\multicolumn{%s}{c}{%s} \\\ \n" %(nb_col,multi_lines_dep)
                lines[x] = lines[x].strip() + to_add

    if new_row != False and multicolumn == None:
        for x, line in enumerate(lines):
            if x == 11:
                lines[x] = lines[x].strip() + new_row_[0]

    if new_row == False and multicolumn != None:
        for x, line in enumerate(lines):
            multi = """
            \n\\\[-1.8ex]
            """
            for key, value in multicolumn.items():
                to_add = "&\multicolumn{%s}{c}{%s}" %(value, key)
                multi+= to_add
            multi+="\\\\\n"
            if x == 10:
                lines[x] = lines[x].strip() + multi

    if new_row != False and multicolumn != None:
        for x, line in enumerate(lines):
            multi = """
            \n\\\[-1.8ex]
            """
            for key, value in multicolumn.items():
                to_add = "&\multicolumn{%s}{c}{%s}" %(value, key)
                multi+= to_add
            multi+="\\\\\n"
            if x == 10:
                lines[x] = lines[x].strip() + multi
            if x == 11:
                lines[x] = lines[x].strip() + new_row_[0]

    ### Add header
    len_line = len(lines)
    for x, line in enumerate(lines):
        if x ==1:
            if jupyter_preview:
                header= "\documentclass[preview]{standalone} \n\\usepackage[utf8]{inputenc}\n" \
            "\\usepackage{booktabs,caption,threeparttable, siunitx, adjustbox}\n\n" \
            "\\begin{document}"
            else:
                header= "\documentclass[12pt]{article} \n\\usepackage[utf8]{inputenc}\n" \
            "\\usepackage{booktabs,caption,threeparttable, siunitx, adjustbox}\n\n" \
            "\\begin{document}"

            lines[x] =  header

        if x == len_line- 1:
            footer = "\n\n\\end{document}"
            lines[x]  =  lines[x].strip() + footer

    with open(table_out, "w") as f:
        for line in lines:
            f.write(line)

    #### rename variables
    with open(table_out, 'r') as file:
        lines = file.read()

        # rename from dictionary
        #for i in data['to_rename']:
            #lines = lines.replace(i['old'],i['new'])
     #       lines = re.sub('\b{}\b'.format(i['old']), i['new'], lines)


        #### very risky
        lines = lines.replace('(0.000)',
                              '')



        ### Should be at the end of the regex
        ### Convert : to time, but not for the title
        lines = lines.replace(':', ' \\times ')
        lines = lines.replace('variable \\times ',
         'variable:')


    # Write the file out again
    with open(table_out, 'w') as file:
        file.write(lines)

    ### Remove empty lines (usually due to fixed effect)
    with open(table_out, 'r') as f:
        lines = f.readlines()

    list_lines_to_remove = []
    for x, line in enumerate(lines):
        test = bool(re.search(r'\d', line))
        if test == False:
            charRe = re.compile(r'^[\W]+$')
            string = charRe.search(line)
            if bool(string) and not line.isspace():
                list_lines_to_remove.append(x -1)
                list_lines_to_remove.append(x)

    with open(table_out, "w") as f:
        for x, line in enumerate(lines):
            if x not in list_lines_to_remove:
                f.write(line)


    #### rename variables -> use regex to avoid replace substring not 100% matched
    with open(table_out, 'r') as f:
        lines = f.readlines()

    for i in data['to_rename']:
        for x, line in enumerate(lines):
            regex_ = i['old'].replace('\_','\\\_')
            matches = re.search(r'^{}$'.format(regex_), line)
            if matches:
                print(matches)
                lines[x] = lines[x].replace(i['old'],i['new'])
            else:

                matches = re.search(r'^{}'.format(regex_), line)
            if matches:
                lines[x] = lines[x].replace(i['old'],i['new'])
            else:
                
                matches = re.search(r'{}\s'.format(regex_), line)
            if matches:
                lines[x] = lines[x].replace(i['old'],i['new'])
            else:
                ### Try when the variable is in log
                matches = re.search(r'log\({}\)'.format(regex_), line)
            
            if matches:
                lines[x] = lines[x].replace(i['old'],i['new'])


    with open(table_out, "w") as f:
        for line in lines:
            f.write(line)



    #### Reorder variables
    if reorder_var != None:

        with open(table_out, 'r') as f:
            lines = f.readlines()

        regex = r"\((\d+)\)"
        found = False
        for x, line in enumerate(lines):
            matches = re.search(regex, line)
            if matches  and found != True:
                found = True
                coef_rows = x + 2
        top_text = lines[:coef_rows]
        vars_text = lines[coef_rows:]

        regex = r"\((\d+)\)"
        found = False
        for x, line in enumerate(lines):
            matches = re.search(regex, line)
            if matches  and found != True:
                found = True
                coef_rows = x + 2
        top_text = lines[:coef_rows]
        vars_text = lines[coef_rows:]

        regex= r"\((\d+)"
        count_var = -1
        for x, line in enumerate(vars_text):
            matches = re.search(regex, line)
            if matches:
                count_var+=1
        bottom_text = vars_text[count_var*2:]
        count_var += 2

        new_order = [None] * count_var
        for key, value in reorder_var.items():
            new_order[value] = key * 2
        for i in range(0, count_var * 2, 2):
            if i not in new_order:
                position = next(i for i, j in enumerate(new_order) if j == None)
                new_order[position] = i
        reorder_full = []
        for order in new_order:
            reorder_full.append(vars_text[order])
            reorder_full.append(vars_text[order+1])
        new_table = top_text + reorder_full[:-1] + bottom_text[3:]

        with open(table_out, "w") as f:
            for x, line in enumerate(new_table):
                f.write(line)


    ### add table #
    if table_nte != None:
        with open(table_out, 'r') as f:
            lines = f.readlines()


        for x, line in enumerate(lines):
            adjusted = bool(re.search(r"end{adjustbox}",
                              line))

            if adjusted:
                lines[x] = lines[x].strip() + "\n\\begin{0} \n \\small \n \\item \\\\ \n{1} \n\\end{2}\n".format(
                "{tablenotes}",
                table_nte,
                "{tablenotes}")

        with open(table_out, "w") as f:
            for line in lines:
                f.write(line)

    if jupyter_preview:
        f = open('{}/table_{}.tex'.format(folder,table_number))
        r = tex2pix.Renderer(f, runbibtex=False)
        r.mkpdf('{}/table_{}.pdf'.format(folder,table_number))
        img = WImage(filename='{}/table_{}.pdf'.format(folder,table_number),
         resolution = resolution)
        return display(img)

    ### Add Adjust box
