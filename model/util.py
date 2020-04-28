from rdkit import Chem
import pymol
import tempfile, os
import numpy as np

def render_pymol(mol, img_file, width=300, height=200):
    """ Render rdkit molecule with a pymol ball & stick representation """

    target = "all"
    pymol.cmd.delete(target)
    pymol.cmd.set("max_threads", 4)
    pymol.cmd.viewport(width, height)
    pymol.cmd.bg_color(color="white")

    # Save the file
    fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    Chem.MolToPDBFile(mol, temp_path)
    pymol.cmd.load(temp_path)
    os.close(fd)

    # Representation
    pymol.cmd.color("gray40", target)
    pymol.util.cnc()
    pymol.cmd.show_as("spheres", target)
    pymol.cmd.show("licorice", target)
    pymol.cmd.set("sphere_scale", 0.25)
    pymol.cmd.orient()
    pymol.cmd.zoom(target, complete=1)
    pymol.cmd.png(img_file, ray=1)
    return
