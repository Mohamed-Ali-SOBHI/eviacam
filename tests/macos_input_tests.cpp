// SPDX-License-Identifier: GPL-3.0-or-later
#include "macos/mouseevents.h"
#include <iostream>
#include <stdexcept>
static void require(bool value,const char* reason) { if(!value) throw std::runtime_error(reason); }
int main() {
    try {
        eviacam::MacMouseEvents mouse;
        auto check=[&](long x,long y,int flags,double time,CGEventType type,CGPoint position,int count=0) {
            CGEventRef event=mouse.Create(x,y,flags,CGPointMake(100,100),time);
            require(event!=nullptr,"event allocation");
            const bool valid=CGEventGetType(event)==type && CGPointEqualToPoint(CGEventGetLocation(event),position) &&
                (!count || CGEventGetIntegerValueField(event,kCGMouseEventClickState)==count);
            CFRelease(event);
            require(valid,"Quartz event type, position or click count");
        };
        check(10,-5,0,1,kCGEventMouseMoved,CGPointMake(110,95));
        check(0,0,2,2,kCGEventLeftMouseDown,CGPointMake(100,100),1);
        check(10,20,0,2.1,kCGEventLeftMouseDragged,CGPointMake(110,120));
        check(0,0,4,2.2,kCGEventLeftMouseUp,CGPointMake(100,100),1);
        check(0,0,2,2.3,kCGEventLeftMouseDown,CGPointMake(100,100),2);
        check(0,0,4,2.4,kCGEventLeftMouseUp,CGPointMake(100,100),2);
        check(0,0,2,4,kCGEventLeftMouseDown,CGPointMake(100,100),1);
        check(0,0,4,4.1,kCGEventLeftMouseUp,CGPointMake(100,100),1);
        check(0,0,8,5,kCGEventRightMouseDown,CGPointMake(100,100));
        check(10,20,0,5.1,kCGEventRightMouseDragged,CGPointMake(110,120));
        check(0,0,16,5.2,kCGEventRightMouseUp,CGPointMake(100,100));
        check(0,0,32,6,kCGEventOtherMouseDown,CGPointMake(100,100));
        check(10,20,0,6.1,kCGEventOtherMouseDragged,CGPointMake(110,120));
        check(0,0,64,6.2,kCGEventOtherMouseUp,CGPointMake(100,100));
        check(320,240,0x8000,7,kCGEventMouseMoved,CGPointMake(320,240));
        require(mouse.Create(0,0,123,CGPointZero,8)==nullptr,"invalid command accepted");
        std::cout<<"PASS: native Quartz movement, drag/release, double-click reset and buttons; no events posted\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
