import pathlib
import os
file = pathlib.Path('build/latex/mip_project.tex')
print(f"cleaning {file.name}")

with open(file, 'r') as fs:
    use_package_lines = []
    other_lines = []
    first_use_package_idx = None
    line_num = 0
    start_doc = False
    end_doc = False
    documentclass = False
    for line in fs:
        if line.startswith('\\usepackage'):
            use_package_lines.append(line)
            if first_use_package_idx is None:
                first_use_package_idx = line_num
        else:
            if line.startswith('\\begin{document}'):
                if not start_doc:
                    start_doc = True
                    other_lines.append(line)

            elif line.startswith('\\documentclass'):
                if not documentclass:
                    documentclass = True
                    other_lines.append(line)

            elif line.startswith('\\end{document}'):
                pass

            else:
                other_lines.append(line)
        line_num += 1

os.remove(file)

with open(file, 'w') as fs:
    line_break = 0
    for idx, line in enumerate(other_lines):
        if idx >= first_use_package_idx:
            line_break = idx
            break
        fs.write(line)

    for line in use_package_lines:
        fs.write(line)

    for line in other_lines[idx:]:
        fs.write(line)

    fs.write('\\end{document}')
