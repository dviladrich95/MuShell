//arenatus_thresh_strip_bend_a_minor_natural_equal_temperament_8_400_330bpm_bass.mp3

#include <TinyStepper_28BYJ_48.h>
#include <EasyNeoPixels.h>

float myTime;
float time_switch_list[] = {0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0,
                          0.0,0.0,0.0};
int note_counter_beg=0;
int note_counter_end=0;

long int mp3_time  = 72870;
int player_delay = 653;
float mp3_time_s  = 72.870;
long int mp3_tot_time = mp3_time+player_delay;
long int real_time = 79404;

float correction_factor = float(real_time)/(mp3_time+player_delay); 
float bps = 330/60;
int qtime_list[]
{
1,2,2,7,7,7,10,11,11,12,15,16,16,16,18,20,21,23,23,23,24,24,26,27,32,34,38,40,42,42,43,43,47,48,48,51,51,51,52,53,55,56,56,57,60,60,62,66,66,66,66,66,67,68,70,70,70,72,74,76,79,80,83,87,89,90,94,95,96,99,101,103,107,108,111,112,117,117,118,121,124,124,125,129,130,132,134,135,137,138,141,142,142,143,146,148,150,151,155,156,159,162,162,163,163,163,166,166,166,169,169,171,173,174,177,178,178,178,179,181,182,183,183,186,188,192,193,194,197,198,199,201,202,210,210,210,212,214,214,216,219,220,223,226,226,228,231,231,232,233,235,235,237,238,239,240,244,245,246,248,248,250,251,255,257,261,262,264,265,265,267,268,270,273,274,275,275,280,280,280,282,285,288,288,288,290,291,291,295,295,296,297,297,298,299,302,302,303,306,306,307,308,310,311,312,312,312,313,313,319,319,321,321,321,322,324,324,326,327,328,329,330,332,332,332,334,334,337,340,340,340,343,344,344,347,347,349,350,351,352,352,355,357,357,358,360,361,363,363,363,365,367,368,371,371,373,375,375,377,378,379,382,384,386,388,388,390,391,395,396
};
int qtime_num=(sizeof(qtime_list) / sizeof(qtime_list[0]));
int last_time_p1=qtime_list[qtime_num-1]+1;
int qpitch_list[]
{
8,12,10,7,10,8,7,10,8,10,7,10,8,10,10,8,10,8,10,7,10,10,7,10,8,8,7,7,8,10,7,7,8,8,10,7,8,10,7,10,7,8,10,7,8,7,8,8,10,8,8,10,7,10,5,7,8,10,8,10,8,10,7,7,8,7,8,8,10,7,10,7,5,8,5,8,7,8,8,7,8,5,8,5,7,5,5,7,5,5,8,5,8,5,8,5,7,5,5,7,5,5,8,7,8,5,7,5,8,5,7,5,3,5,5,7,7,8,3,5,8,3,7,5,3,5,7,5,7,3,5,7,3,7,5,5,3,5,7,5,5,3,7,3,3,7,5,3,3,7,3,5,3,5,5,3,7,3,3,7,5,3,5,2,3,2,3,7,5,2,3,7,5,3,5,2,3,2,3,5,2,3,2,2,2,3,5,2,2,2,5,2,3,3,5,5,2,3,5,5,0,2,5,2,5,2,2,3,3,2,3,2,3,3,5,2,0,5,3,3,5,0,5,3,3,5,2,0,2,3,0,0,0,3,3,0,3,5,2,2,3,5,2,3,0,3,3,0,2,5,3,0,5,5,0,5,2,0,3,2,3,3,3,5,0,2,5,3,2,2
};
int qpitch_num=(sizeof(qpitch_list) / sizeof(qpitch_list[0]));

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
    writeEasyNeoPixel(pitch_led, rb_r[pitch_led]/10,rb_g[pitch_led]/10,rb_b[pitch_led]/10);
    time_switch_list[pitch_led] = myTime;
    }


    for (int i=0; i<12; i++){
      if(myTime-time_switch_list[i]>duration/bps){
        writeEasyNeoPixel(i, LOW);
      }
    }
    note_counter_beg = note_counter_end;
  
  stepper.moveRelativeInSteps(8);
  if((millis()%mp3_tot_time) <1000){
    Serial.print("millis \n");
    Serial.print(millis());
    note_counter_beg=0;
    note_counter_end=0;
    }

}
