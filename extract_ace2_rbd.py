from Bio import PDB


parser = PDB.PDBParser(QUIET=True)


structure = parser.get_structure("6M0J", "6M0J.pdb")


io = PDB.PDBIO()

# Extract chain A (ACE2)
io.set_structure(structure)
io.select = lambda chain: chain.id == "A"
io.save("ace2.pdb")

# Extract chain E (RBD)
io.set_structure(structure)
io.select = lambda chain: chain.id == "E"
io.save("rbd.pdb")

print("Chains extracted successfully to ace2.pdb and rbd.pdb")
