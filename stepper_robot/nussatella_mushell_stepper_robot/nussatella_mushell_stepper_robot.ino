//arenatus_thresh_strip_bend_a_minor_natural_equal_temperament_8_400_330bpm_bass.mp3

#include <TinyStepper_28BYJ_48.h>
#include <EasyNeoPixels.h>

float myTime;
float time_switch_list[] = {0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0};
float time_switch_list2[] = {0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0};
                
int note_counter_beg=0;
int note_counter_end=0;

int note_counter_beg2=0;
int note_counter_end2=0;

long int mp3_time  = 57768;
int player_delay = 653;
float mp3_time_s  = float(mp3_time)/1000;
long int mp3_tot_time = mp3_time+player_delay;
long int real_time = 53234;

float correction_factor = float(real_time)/(mp3_time+player_delay); 
float bps = 120/60;
int qtime_list[]
{
0,0,1,4,5,7,9,9,11,13,13,13,17,18,18,21,22,22,25,27,27,27,29,31,31,31,33,34,36,37,38,40,41,41,43,44,44,46,48,48,49,52,52,52,56,56,56,60,61,63,63,64,67,67,68,71,71,74,75,75,79,79,79,81,81,83,83,85,86,86,87,91,91,92,93,95,95,95,100,100,101,105,105,105,110,112,112,116,118,118
};
int qtime_num=(sizeof(qtime_list) / sizeof(qtime_list[0]));
int last_time_p1=qtime_list[qtime_num-1]+1;

int qtime_list2[]
{
0,0,3,6,8,10,12,12,14,16,18,20,21,23,25,27,30,33,33,36,36,39,39,43,43,47,47,50,50,54,54,58,59,61,61,65,66,69,69,71,73,73,76,77,80,81,83,85,86,89,90,92,95,95,99,100,101,105,106,110,111,116,118
};
int qtime_num2=(sizeof(qtime_list2) / sizeof(qtime_list2[0]));
int last_time_p12=qtime_list2[qtime_num2-1]+1;

int qpitch_list[]
{
9,7,10,9,7,10,7,7,10,9,10,7,7,10,9,7,10,9,7,9,10,7,10,7,9,5,7,9,5,7,9,5,7,9,5,9,7,5,5,4,7,5,7,4,7,5,4,2,5,5,4,2,4,5,2,4,5,2,4,5,4,2,5,2,2,4,2,5,2,2,4,2,2,4,4,4,2,0,4,2,0,0,4,2,0,2,2,0,2,2
};
int qpitch_num=(sizeof(qpitch_list) / sizeof(qpitch_list[0]));

int qpitch_list2[]
{
10,7,10,7,10,7,10,7,10,7,10,7,10,7,10,7,7,7,9,7,9,7,5,7,7,7,5,7,5,4,5,5,4,4,5,5,2,4,2,5,4,2,4,2,4,2,2,4,2,2,4,2,2,2,2,2,0,2,0,2,0,2,0
};
int qpitch_num2=(sizeof(qpitch_list2) / sizeof(qpitch_list2[0]));

int rb_r[]{255,255,255,128,0  ,0  ,0  ,0  ,0  ,128,255,255};
int rb_g[]{0  ,0  ,0  ,0  ,0  ,128,255,255,255,255,255,128};
int rb_b[]{0  ,128,255,255,255,255,255,128,0  ,0  ,0  ,0  };

int duration = 1000;

TinyStepper_28BYJ_48 stepper;

void setup() {
  Serial.begin(9600);

  stepper.connectToPins(8, 9, 10, 11);
  stepper.setSpeedInStepsPerSecond(2048.0/mp3_time_s);
  stepper.setAccelerationInStepsPerSecondPerSecond(512);

    // setup for 12 NeoPixel attached to pin 2
  setupEasyNeoPixels(2, 12);

}

void loop() {
  
  float myTime = millis();
  
  while (qtime_list[note_counter_end]<int(correction_factor*bps*myTime/1000)%last_time_p1){
  note_counter_end += 1;
  }
  for (int i=note_counter_beg; i<note_counter_end; i++){
    int pitch_led = constrain(qpitch_list[i],0,11);
    writeEasyNeoPixel(pitch_led, 0,0,25);
    time_switch_list[pitch_led] = myTime;
    }
//    for (int i=0; i<12; i++){
//      if(myTime-time_switch_list[i]>duration){
//        writeEasyNeoPixel(i, LOW);
//      }
//    }
    note_counter_beg = note_counter_end;

  while (qtime_list2[note_counter_end2]<int(correction_factor*bps*myTime/1000)%last_time_p12){
  note_counter_end2 += 1;
  }
  for (int i=note_counter_beg2; i<note_counter_end2; i++){
    int pitch_led2 = constrain(qpitch_list2[i],0,11);
    writeEasyNeoPixel(pitch_led2, 25,0,0);
    time_switch_list2[pitch_led2] = myTime;
    }
    for (int i=0; i<12; i++){
      if((myTime-time_switch_list2[i]>duration) && (myTime-time_switch_list[i]>duration)){
        writeEasyNeoPixel(i, LOW);
      }
    }
    note_counter_beg2 = note_counter_end2;
  
  stepper.moveRelativeInSteps(8);
  //if((millis()%mp3_tot_time) <1000){
  if(note_counter_beg>=qtime_num-2){
    Serial.print("millis \n");
    Serial.print(millis());
    note_counter_beg=0;
    note_counter_end=0;
    note_counter_beg2=0;
    note_counter_end2=0;
    }

}
