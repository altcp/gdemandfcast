# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# import gc
# 
# import src.functions as f
# 

# %%
def main():

    #UI
    print("Welcome to Automated Door Opening System (ADOS).")
    u1_q1()
    print(" ")
    print(" ")
 
    return None
    


# %%
def u1_q1():

    ans_one = 0

    while(ans_one < 1):


        print("ADOS: Would you like to take a test picture? ")
        input_ans_one = input("Enter Yes (1) / No (2) : ")
        
        try:         
            ans_one = int(input_ans_one)
        except:
            ans_one = 0


        if (ans_one == 1):

            #Button Integration.
            #Lionel, are you able to help ?
        
            picture_taken = f.run_camera()
            

            if(picture_taken == True):
                
                try:

                    if (f.run_model(picture_taken) == True):
                        f.clean_up()
                        print(" ")
                        print(" ")
                        print("Thank You. This completes our Demo. ")

                except:
                    print("ADOS: Error. Kindly Contact Developer. ")
        
        
        elif(ans_one == 2):
            print(" ")
            print(" ")
            print(" ")
        
        else:
            print("Please enter 1 or 2")
            ans_one = 0

    return None


# %%
if __name__ == "__main__":
    
    main()
    gc.collect()
    quit()


