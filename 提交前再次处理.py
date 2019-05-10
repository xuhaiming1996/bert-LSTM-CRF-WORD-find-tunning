def predict(model,filepath_train,filepath_p):
    f_p = open(filepath_p, mode="r", encoding="utf-8")
    fw=open("./data/result.json",mode="w",encoding="utf-8")
    tmp={"text":"", "spo_list":[]}
    num=0
    with open(filepath_train, mode="r", encoding="utf-8") as f:
        words, poses, p = [], [], ""
        for line in f:
            line = line.rstrip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    preds = model.predict(words_raw=words,poses_raw=poses, p_raw=p)
                    subject=[]
                    object=[]
                    for word, pred in zip(words,preds):
                        if pred.find("OBJ")!=-1:
                            object.append(word)
                        elif pred.find("SUB")!=-1:
                            subject.append(word)
                    #     print(word+"_"+pred, end=" ")
                    # print("\n")
                    pre_all, _ = f_p.readline().strip().split("\t")
                    text_dict = eval(pre_all)
                    text=text_dict["text"]
                    subject="".join(subject)
                    object="".join(object)
                    # print("text:",text)
                    # print("spo:",subject,object,p)
                    a = {"object_type": "", "predicate": "", "object": "", "subject_type": "", "subject": ""}
                    a["predicate"] = p
                    a["object"] = object
                    a["subject"] = subject
                    if text==tmp["text"]:
                        tmp["spo_list"].append(a)
                    else:
                        fw.write(json.dumps(tmp, ensure_ascii=False)+"\n")
                        if num%100==0:
                            print("写：",num)
                        num += 1
                        tmp["text"]=text
                        tmp["spo_list"]=[a]
                    words, poses,  p = [], [], ""
            else:
                ls = line.split('\t')
                if len(ls) == 1:
                    p = ls[0]
                else:
                    # print(line)
                    word, pos, = ls[0], ls[1]
                    words += [word]
                    poses += [pos]

    fw.write(json.dumps(tmp, ensure_ascii=False))

    fw.close()
    print(num)
