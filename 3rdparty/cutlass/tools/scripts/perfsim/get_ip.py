#!/home/utils/Python-3.6.1/bin/python3.6
import os, stat
import subprocess
import re


def get_ip(instr):
    # ip is present as a comment => split based on # and use the last part

    split_instr = instr.split("#")
    z = re.findall(r"\[.*?]", split_instr[-1])
    ip = re.findall(r"[a-f0-9]+", z[1])[0]

    return ip

def create_script(cmds, filename):
    f = open(filename)
    f.write(cmds)
    os.chmod(filename, stat.S_IRWXU)

def generate_phasealyze_spec(mainloop_start_ip, mainloop_end_ip):
    # Find all the morph files
    cmd = "find . | grep morphism/morphology.csv > morph_files.txt"
    create_script(cmd, "morph_files.sh")
    subprocess.call(["./morph_files.sh"])
    spec =  open("prramani.spec")

    
def get_workload_name(cs_files):
    workloads = []

    for filename in cs_files:
        split_path = filename.split("/")
        index = 0
        for folder_name in split_path:
            if "run" in folder_name and "dir.0" in folder_name :
                break
            else
                index += 1

        workload_id = split_path[index+1].split(".")[0]

        state_yml = "./perfsim/apic_capture/run.A.dir.0/{0}/state.yml".format(workload_id)
        workload_name = None
        with open(state_yml) as statefile:
            yml = statefile.readlines()
            for line in reversed(yml):
                if "workload" in line :
                    workload_name = line.split(":")[1]
                    break
                    
        workloads.append(workload_name)
    return workloads

def main():
    cmd = "find . | grep decodecp.txt > cs_files.txt"
    create_script(cmd, "cs_files.sh")
    subprocess.call(["./cs_files.sh"])

    with open("cs_files.txt") as f:
        
        files = f.readlines()
        workload_names = get_workload_name(files)

        cs_file_idx = 0
        for cs in files:
            cs = cs.strip()
            
            branch_instruction = ""
            branch_ip = ""
            label_next_ip = ""
            
            with open(cs) as compute_shader:
                instructions = compute_shader.readlines()

        
                # STEP 1 find the last HMMA, Branch instruction
                for instruction in reversed(instructions):
                    if "BRA" in instruction : 
                        branch_instruction = instruction
                    
                    if "FP64in" in cs:
                        if "DMMA" in instruction:
                            break
                    elif "HMMA" in instruction:
                        break

                # Step 2 : get the branch IP and the branch label (regex)
                #z = re.findall(r"\[.*?]", branch_instruction)
                #branch_ip = re.findall(r"[a-f0-9]+", z[1])[0]
        
                branch_ip = get_ip(branch_instruction)

                tmp = re.findall(r"BRA[\t\s]+[A-Za-z0-9]+", branch_instruction)
                branch_label = tmp[0].split(" ")[-1]

                mainloop_start = [None, None]
                for instruction in reversed(instructions):
                    if str(branch_label) + ":" in instruction :
                        mainloop_start[0] = instruction
                        break;
                    else:
                        mainloop_start[1] = instruction
                
                mainloop_ip = get_ip(mainloop_start[1])

                print(cs)
                print(workload_names[cs_file_idx])
                print(mainloop_ip)
                print(branch_ip)
                #print(branch_instruction)
                #print(branch_label)
                #print(str(branch_label) + ":")

            cs_file_idx += 1

if __name__ == "__main__":
    main()
