from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import json
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.015, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # json
    jsonString = '{'
    img_num = 0
    img_filer = os.listdir(args.img)
    for img_file in img_filer:
        img = os.path.join(args.img, img_file)
        result = inference_detector(model, img)
        new_result = result[0][0]

        point_num = 0
        bbox_str = ""
        if(img_num==0):
            bbox_str = '"'+img_file+'":['
            img_num +=1
        else:
            bbox_str = ',"'+img_file+'":['
        for result_arr in new_result:
          if(result_arr[4]>=args.score_thr):  
            img_score = ('%.5f'%result_arr[4])
            if(point_num==0):
                bbox_str = bbox_str+"["+str(round(result_arr[0]))+","+str(round(result_arr[1]))+","+str(round(result_arr[2]))+","+str(round(result_arr[3]))+","+img_score+"]"
            else:
                bbox_str = bbox_str+",["+str(round(result_arr[0]))+","+str(round(result_arr[1]))+","+str(round(result_arr[2]))+","+str(round(result_arr[3]))+","+img_score+"]"
            point_num+=1
        # print(bbox_str)
        jsonString = jsonString+bbox_str
        jsonString = jsonString+"]"

    jsonString = jsonString+"}"	

    jsonString = json.loads(jsonString)
    json_file = open('output.json', "w")
    json.dump(jsonString, json_file)
    json_file.close()
    # show the results
    # show_result_pyplot(model, args.img, new_result, score_thr=args.score_thr)



if __name__ == '__main__':
    main()
