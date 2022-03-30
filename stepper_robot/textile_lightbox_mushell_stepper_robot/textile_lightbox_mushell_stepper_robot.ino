//conus_textile_lightbox_thresh_strip_dots_balafon_1_8_400_300bpm_piano.mp3

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
float bps = 300/60;
int qtime_list[]
{
1,1,8,8,11,11,24,24,25,27,29,29,30,30,30,41,56,57,58,59,64,64,68,73,73,73,79,81,82,87,87,87,91,92,93,99,105,110,110,112,112,116,118,127,130,134,134,143,149,150,152,162,163,166,167,168,170,172,177,180,187,189,190,196,199,199,205,207,208,211,212,214,216,222,226,226,231,232,239,241,241,242,248,249,249,252,252,253,255,256,259,262,263,268,271,272,272,274,280,280,280,281,282,284,288,291,298,299,299,301,302,307,307,311,313,314,319,321,323,325,337,338,340,341,343,346,347,353,353,353,355,355,355,361,361,365,369,373,374,376,378,381,383,383,384,389
};
int qtime_num=(sizeof(qtime_list) / sizeof(qtime_list[0]));
int last_time_p1=qtime_list[qtime_num-1]+1;
int qpitch_list[]
{
4,7,4,9,7,7,4,5,4,7,5,2,5,5,9,4,7,7,5,4,5,2,7,9,5,9,7,4,2,2,2,4,5,7,4,7,9,4,7,5,7,2,2,7,4,2,5,9,7,7,4,5,4,10,7,2,7,5,9,5,2,7,4,9,5,7,7,5,2,2,5,5,2,7,4,9,2,0,4,7,2,5,2,2,7,4,10,9,2,5,9,5,2,4,2,2,7,4,4,5,9,2,7,7,7,7,4,4,7,5,7,2,4,4,9,5,7,7,4,10,10,2,7,4,9,5,5,2,4,2,4,7,7,9,10,2,2,7,9,4,4,4,2,7,4,7
};
int qpitch_num=(sizeof(qpitch_list) / sizeof(qpitch_list[0]));

int rb_r[]{255,255,255,128,0  ,0  ,0  ,0  ,0  ,128,255,255};
int rb_g[]{0  ,0  ,0  ,0  ,0  ,128,255,255,255,255,255,128};
int rb_b[]{0  ,128,255,255,255,255,255,128,0  ,0  ,0  ,0  };

int duration = 1000;

TinyStepper_28BYJ_48 stepper;

void setup() {
  Serial.begin(9600);

  stepper.connectToPins(11, 10, 9, 8);
  stepper.setSpeedInStepsPerSecond(2048.0/mp3_time_s);
  stepper.setAccelerationInStepsPerSecondPerSecond(512);

    // setup for 12 NeoPixel attached to pin 2
  setupEasyNeoPixels(2, 12);

}

void loop() {
  
  float myTime = millis();
  //while (qtime_list[note_counter_end]<int(correction_factor*bps*myTime/1000)%last_time_p1){
  while (qtime_list[note_counter_end]<int(bps*myTime/1000)%last_time_p1){
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
  if(note_counter_beg>=qtime_num-2){
    Serial.print("millis \n");
    Serial.print(millis());
    note_counter_beg=0;
    note_counter_end=0;
    }

}
