/*******************************************************************************************************************
Copyright (c) 2023 Cycling '74

The code that Max generates automatically and that end users are capable of
exporting and using, and any associated documentation files (the “Software”)
is a work of authorship for which Cycling '74 is the author and owner for
copyright purposes.

This Software is dual-licensed either under the terms of the Cycling '74
License for Max-Generated Code for Export, or alternatively under the terms
of the General Public License (GPL) Version 3. You may use the Software
according to either of these licenses as it is most appropriate for your
project on a case-by-case basis (proprietary or not).

A) Cycling '74 License for Max-Generated Code for Export

A license is hereby granted, free of charge, to any person obtaining a copy
of the Software (“Licensee”) to use, copy, modify, merge, publish, and
distribute copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The Software is licensed to Licensee for all uses that do not include the sale,
sublicensing, or commercial distribution of software that incorporates this
source code. This means that the Licensee is free to use this software for
educational, research, and prototyping purposes, to create musical or other
creative works with software that incorporates this source code, or any other
use that does not constitute selling software that makes use of this source
code. Commercial distribution also includes the packaging of free software with
other paid software, hardware, or software-provided commercial services.

For entities with UNDER $200k in annual revenue or funding, a license is hereby
granted, free of charge, for the sale, sublicensing, or commercial distribution
of software that incorporates this source code, for as long as the entity's
annual revenue remains below $200k annual revenue or funding.

For entities with OVER $200k in annual revenue or funding interested in the
sale, sublicensing, or commercial distribution of software that incorporates
this source code, please send inquiries to licensing@cycling74.com.

The above copyright notice and this license shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Please see
https://support.cycling74.com/hc/en-us/articles/10730637742483-RNBO-Export-Licensing-FAQ
for additional information

B) General Public License Version 3 (GPLv3)
Details of the GPLv3 license can be found at: https://www.gnu.org/licenses/gpl-3.0.html
*******************************************************************************************************************/

#ifdef RNBO_LIB_PREFIX
#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)
#define RNBO_LIB_INCLUDE(X) STR(RNBO_LIB_PREFIX/X)
#else
#define RNBO_LIB_INCLUDE(X) #X
#endif // RNBO_LIB_PREFIX
#ifdef RNBO_INJECTPLATFORM
#define RNBO_USECUSTOMPLATFORM
#include RNBO_INJECTPLATFORM
#endif // RNBO_INJECTPLATFORM

#include RNBO_LIB_INCLUDE(RNBO_Common.h)
#include RNBO_LIB_INCLUDE(RNBO_AudioSignal.h)

namespace RNBO {


#define trunc(x) ((Int)(x))
#define autoref auto&

#if defined(__GNUC__) || defined(__clang__)
    #define RNBO_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RNBO_RESTRICT __restrict
#endif

#define FIXEDSIZEARRAYINIT(...) { }

template <class ENGINE = INTERNALENGINE> class rnbomatic : public PatcherInterfaceImpl {

friend class EngineCore;
friend class Engine;
friend class MinimalEngine<>;
public:

rnbomatic()
: _internalEngine(this)
{
}

~rnbomatic()
{
    deallocateSignals();
}

Index getNumMidiInputPorts() const {
    return 1;
}

void processMidiEvent(MillisecondTime time, int port, ConstByteArray data, Index length) {
    this->updateTime(time, (ENGINE*)nullptr);
    this->p_01_midihandler(data[0] & 240, (data[0] & 15) + 1, port, data, length);
}

Index getNumMidiOutputPorts() const {
    return 0;
}

void process(
    const SampleValue * const* inputs,
    Index numInputs,
    SampleValue * const* outputs,
    Index numOutputs,
    Index n
) {
    this->vs = n;
    this->updateTime(this->getEngine()->getCurrentTime(), (ENGINE*)nullptr, true);
    SampleValue * out1 = (numOutputs >= 1 && outputs[0] ? outputs[0] : this->dummyBuffer);
    SampleValue * out2 = (numOutputs >= 2 && outputs[1] ? outputs[1] : this->dummyBuffer);
    const SampleValue * in1 = (numInputs >= 1 && inputs[0] ? inputs[0] : this->zeroBuffer);
    this->p_01_perform(in1, this->zeroBuffer, this->signals[0], this->dummyBuffer, n);
    this->signalforwarder_01_perform(this->signals[0], out2, n);
    this->signalforwarder_02_perform(this->signals[0], out1, n);
    this->stackprotect_perform(n);
    this->globaltransport_advance();
    this->advanceTime((ENGINE*)nullptr);
    this->audioProcessSampleCount += this->vs;
}

void prepareToProcess(number sampleRate, Index maxBlockSize, bool force) {
    RNBO_ASSERT(this->_isInitialized);

    if (this->maxvs < maxBlockSize || !this->didAllocateSignals) {
        Index i;

        for (i = 0; i < 1; i++) {
            this->signals[i] = resizeSignal(this->signals[i], this->maxvs, maxBlockSize);
        }

        this->globaltransport_tempo = resizeSignal(this->globaltransport_tempo, this->maxvs, maxBlockSize);
        this->globaltransport_state = resizeSignal(this->globaltransport_state, this->maxvs, maxBlockSize);
        this->zeroBuffer = resizeSignal(this->zeroBuffer, this->maxvs, maxBlockSize);
        this->dummyBuffer = resizeSignal(this->dummyBuffer, this->maxvs, maxBlockSize);
        this->didAllocateSignals = true;
    }

    const bool sampleRateChanged = sampleRate != this->sr;
    const bool maxvsChanged = maxBlockSize != this->maxvs;
    const bool forceDSPSetup = sampleRateChanged || maxvsChanged || force;

    if (sampleRateChanged || maxvsChanged) {
        this->vs = maxBlockSize;
        this->maxvs = maxBlockSize;
        this->sr = sampleRate;
        this->invsr = 1 / sampleRate;
    }

    this->globaltransport_dspsetup(forceDSPSetup);
    this->p_01->prepareToProcess(sampleRate, maxBlockSize, force);

    if (sampleRateChanged)
        this->onSampleRateChanged(sampleRate);
}

number msToSamps(MillisecondTime ms, number sampleRate) {
    return ms * sampleRate * 0.001;
}

MillisecondTime sampsToMs(SampleIndex samps) {
    return samps * (this->invsr * 1000);
}

Index getNumInputChannels() const {
    return 1;
}

Index getNumOutputChannels() const {
    return 2;
}

DataRef* getDataRef(DataRefIndex index)  {
    switch (index) {
    case 0:
        {
        return addressOf(this->RNBODefaultSinus);
        break;
        }
    default:
        {
        return nullptr;
        }
    }
}

DataRefIndex getNumDataRefs() const {
    return 1;
}

void processDataViewUpdate(DataRefIndex index, MillisecondTime time) {
    this->p_01->processDataViewUpdate(index, time);
}

void initialize() {
    RNBO_ASSERT(!this->_isInitialized);

    this->RNBODefaultSinus = initDataRef(
        this->RNBODefaultSinus,
        this->dataRefStrings->name0,
        true,
        this->dataRefStrings->file0,
        this->dataRefStrings->tag0
    );

    this->assign_defaults();
    this->applyState();
    this->RNBODefaultSinus->setIndex(0);
    this->initializeObjects();
    this->allocateDataRefs();
    this->startup();
    this->_isInitialized = true;
}

void setParameterValue(ParameterIndex index, ParameterValue v, MillisecondTime time) {
    RNBO_UNUSED(v);
    this->updateTime(time, (ENGINE*)nullptr);

    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            this->p_01->setParameterValue(index, v, time);

        break;
        }
    }
}

void processParameterEvent(ParameterIndex index, ParameterValue value, MillisecondTime time) {
    this->setParameterValue(index, value, time);
}

void processParameterBangEvent(ParameterIndex index, MillisecondTime time) {
    this->setParameterValue(index, this->getParameterValue(index), time);
}

void processNormalizedParameterEvent(ParameterIndex index, ParameterValue value, MillisecondTime time) {
    this->setParameterValueNormalized(index, value, time);
}

ParameterValue getParameterValue(ParameterIndex index)  {
    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->getParameterValue(index);

        return 0;
        }
    }
}

ParameterIndex getNumSignalInParameters() const {
    return 0;
}

ParameterIndex getNumSignalOutParameters() const {
    return 0;
}

ParameterIndex getNumParameters() const {
    return 0 + this->p_01->getNumParameters();
}

ConstCharPointer getParameterName(ParameterIndex index) const {
    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->getParameterName(index);

        return "bogus";
        }
    }
}

ConstCharPointer getParameterId(ParameterIndex index) const {
    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->getParameterId(index);

        return "bogus";
        }
    }
}

void getParameterInfo(ParameterIndex index, ParameterInfo * info) const {
    {
        switch (index) {
        default:
            {
            index -= 0;

            if (index < this->p_01->getNumParameters())
                this->p_01->getParameterInfo(index, info);

            break;
            }
        }
    }
}

ParameterValue applyStepsToNormalizedParameterValue(ParameterValue normalizedValue, int steps) const {
    if (steps == 1) {
        if (normalizedValue > 0) {
            normalizedValue = 1.;
        }
    } else {
        ParameterValue oneStep = (number)1. / (steps - 1);
        ParameterValue numberOfSteps = rnbo_fround(normalizedValue / oneStep * 1 / (number)1) * (number)1;
        normalizedValue = numberOfSteps * oneStep;
    }

    return normalizedValue;
}

ParameterValue convertToNormalizedParameterValue(ParameterIndex index, ParameterValue value) const {
    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->convertToNormalizedParameterValue(index, value);

        return value;
        }
    }
}

ParameterValue convertFromNormalizedParameterValue(ParameterIndex index, ParameterValue value) const {
    value = (value < 0 ? 0 : (value > 1 ? 1 : value));

    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->convertFromNormalizedParameterValue(index, value);

        return value;
        }
    }
}

ParameterValue constrainParameterValue(ParameterIndex index, ParameterValue value) const {
    switch (index) {
    default:
        {
        index -= 0;

        if (index < this->p_01->getNumParameters())
            return this->p_01->constrainParameterValue(index, value);

        return value;
        }
    }
}

void processNumMessage(MessageTag tag, MessageTag objectId, MillisecondTime time, number payload) {
    RNBO_UNUSED(objectId);
    this->updateTime(time, (ENGINE*)nullptr);
    this->p_01->processNumMessage(tag, objectId, time, payload);
}

void processListMessage(
    MessageTag tag,
    MessageTag objectId,
    MillisecondTime time,
    const list& payload
) {
    RNBO_UNUSED(objectId);
    this->updateTime(time, (ENGINE*)nullptr);
    this->p_01->processListMessage(tag, objectId, time, payload);
}

void processBangMessage(MessageTag tag, MessageTag objectId, MillisecondTime time) {
    RNBO_UNUSED(objectId);
    this->updateTime(time, (ENGINE*)nullptr);
    this->p_01->processBangMessage(tag, objectId, time);
}

MessageTagInfo resolveTag(MessageTag tag) const {
    switch (tag) {

    }

    auto subpatchResult_0 = this->p_01->resolveTag(tag);

    if (subpatchResult_0)
        return subpatchResult_0;

    return "";
}

MessageIndex getNumMessages() const {
    return 0;
}

const MessageInfo& getMessageInfo(MessageIndex index) const {
    switch (index) {

    }

    return NullMessageInfo;
}

protected:

class RNBOSubpatcher_23 : public PatcherInterfaceImpl {
    
    friend class rnbomatic;
    
    public:
    
    RNBOSubpatcher_23()
    {}
    
    ~RNBOSubpatcher_23()
    {
        deallocateSignals();
    }
    
    Index getNumMidiInputPorts() const {
        return 1;
    }
    
    void processMidiEvent(MillisecondTime time, int port, ConstByteArray data, Index length) {
        this->updateTime(time, (ENGINE*)nullptr);
        this->ctlin_01_midihandler(data[0] & 240, (data[0] & 15) + 1, port, data, length);
    }
    
    Index getNumMidiOutputPorts() const {
        return 0;
    }
    
    void process(
        const SampleValue * const* inputs,
        Index numInputs,
        SampleValue * const* outputs,
        Index numOutputs,
        Index n
    ) {
        this->vs = n;
        this->updateTime(this->getEngine()->getCurrentTime(), (ENGINE*)nullptr, true);
        SampleValue * out1 = (numOutputs >= 1 && outputs[0] ? outputs[0] : this->dummyBuffer);
        SampleValue * out2 = (numOutputs >= 2 && outputs[1] ? outputs[1] : this->dummyBuffer);
        const SampleValue * in1 = (numInputs >= 1 && inputs[0] ? inputs[0] : this->zeroBuffer);
        const SampleValue * in2 = (numInputs >= 2 && inputs[1] ? inputs[1] : this->zeroBuffer);
        this->noise_tilde_01_perform(this->signals[0], n);
    
        this->cycle_tilde_01_perform(
            in1,
            this->cycle_tilde_01_phase_offset,
            this->signals[1],
            this->signals[2],
            n
        );
    
        this->numbertilde_01_perform(in1, this->dummyBuffer, n);
        this->saw_tilde_01_perform(in1, this->saw_tilde_01_syncPhase, this->signals[3], this->signals[4], n);
    
        this->rect_tilde_01_perform(
            in1,
            in2,
            this->rect_tilde_01_syncPhase,
            this->signals[5],
            this->signals[6],
            n
        );
    
        this->tri_tilde_01_perform(
            in1,
            in2,
            this->tri_tilde_01_syncPhase,
            this->signals[7],
            this->signals[8],
            n
        );
    
        this->numbertilde_02_perform(in2, this->dummyBuffer, n);
    
        this->rect_tilde_02_perform(
            in1,
            in2,
            this->rect_tilde_02_syncPhase,
            this->signals[9],
            this->signals[10],
            n
        );
    
        this->selector_01_perform(
            this->selector_01_onoff,
            this->signals[0],
            this->signals[1],
            this->signals[3],
            this->signals[7],
            this->signals[5],
            this->signals[9],
            out1,
            n
        );
    
        this->ip_01_perform(this->signals[9], n);
    
        this->selector_02_perform(
            this->selector_02_onoff,
            this->signals[9],
            this->signals[2],
            this->signals[4],
            this->signals[8],
            this->signals[6],
            this->signals[10],
            out2,
            n
        );
    
        this->stackprotect_perform(n);
        this->audioProcessSampleCount += this->vs;
    }
    
    void prepareToProcess(number sampleRate, Index maxBlockSize, bool force) {
        RNBO_ASSERT(this->_isInitialized);
    
        if (this->maxvs < maxBlockSize || !this->didAllocateSignals) {
            Index i;
    
            for (i = 0; i < 11; i++) {
                this->signals[i] = resizeSignal(this->signals[i], this->maxvs, maxBlockSize);
            }
    
            this->ip_01_sigbuf = resizeSignal(this->ip_01_sigbuf, this->maxvs, maxBlockSize);
            this->zeroBuffer = resizeSignal(this->zeroBuffer, this->maxvs, maxBlockSize);
            this->dummyBuffer = resizeSignal(this->dummyBuffer, this->maxvs, maxBlockSize);
            this->didAllocateSignals = true;
        }
    
        const bool sampleRateChanged = sampleRate != this->sr;
        const bool maxvsChanged = maxBlockSize != this->maxvs;
        const bool forceDSPSetup = sampleRateChanged || maxvsChanged || force;
    
        if (sampleRateChanged || maxvsChanged) {
            this->vs = maxBlockSize;
            this->maxvs = maxBlockSize;
            this->sr = sampleRate;
            this->invsr = 1 / sampleRate;
        }
    
        this->noise_tilde_01_dspsetup(forceDSPSetup);
        this->cycle_tilde_01_dspsetup(forceDSPSetup);
        this->numbertilde_01_dspsetup(forceDSPSetup);
        this->saw_tilde_01_dspsetup(forceDSPSetup);
        this->tri_tilde_01_dspsetup(forceDSPSetup);
        this->numbertilde_02_dspsetup(forceDSPSetup);
        this->ip_01_dspsetup(forceDSPSetup);
    
        if (sampleRateChanged)
            this->onSampleRateChanged(sampleRate);
    }
    
    number msToSamps(MillisecondTime ms, number sampleRate) {
        return ms * sampleRate * 0.001;
    }
    
    MillisecondTime sampsToMs(SampleIndex samps) {
        return samps * (this->invsr * 1000);
    }
    
    Index getNumInputChannels() const {
        return 2;
    }
    
    Index getNumOutputChannels() const {
        return 2;
    }
    
    void setParameterValue(ParameterIndex index, ParameterValue v, MillisecondTime time) {
        this->updateTime(time, (ENGINE*)nullptr);
    
        switch (index) {
        case 0:
            {
            this->param_01_value_set(v);
            break;
            }
        }
    }
    
    void processParameterEvent(ParameterIndex index, ParameterValue value, MillisecondTime time) {
        this->setParameterValue(index, value, time);
    }
    
    void processParameterBangEvent(ParameterIndex index, MillisecondTime time) {
        this->setParameterValue(index, this->getParameterValue(index), time);
    }
    
    void processNormalizedParameterEvent(ParameterIndex index, ParameterValue value, MillisecondTime time) {
        this->setParameterValueNormalized(index, value, time);
    }
    
    ParameterValue getParameterValue(ParameterIndex index)  {
        switch (index) {
        case 0:
            {
            return this->param_01_value;
            }
        default:
            {
            return 0;
            }
        }
    }
    
    ParameterIndex getNumSignalInParameters() const {
        return 0;
    }
    
    ParameterIndex getNumSignalOutParameters() const {
        return 0;
    }
    
    ParameterIndex getNumParameters() const {
        return 1;
    }
    
    ConstCharPointer getParameterName(ParameterIndex index) const {
        switch (index) {
        case 0:
            {
            return "mode";
            }
        default:
            {
            return "bogus";
            }
        }
    }
    
    ConstCharPointer getParameterId(ParameterIndex index) const {
        switch (index) {
        case 0:
            {
            return "osc.analog/mode";
            }
        default:
            {
            return "bogus";
            }
        }
    }
    
    void getParameterInfo(ParameterIndex index, ParameterInfo * info) const {
        {
            switch (index) {
            case 0:
                {
                info->type = ParameterTypeNumber;
                info->initialValue = 2;
                info->min = 0;
                info->max = 5;
                info->exponent = 1;
                info->steps = 6;
                static const char * eVal0[] = {"noise", "sine", "saw", "triangle", "square", "pulse"};
                info->enumValues = eVal0;
                info->debug = false;
                info->saveable = true;
                info->transmittable = true;
                info->initialized = true;
                info->visible = true;
                info->displayName = "";
                info->unit = "";
                info->ioType = IOTypeUndefined;
                info->signalIndex = INVALID_INDEX;
                break;
                }
            }
        }
    }
    
    ParameterValue applyStepsToNormalizedParameterValue(ParameterValue normalizedValue, int steps) const {
        if (steps == 1) {
            if (normalizedValue > 0) {
                normalizedValue = 1.;
            }
        } else {
            ParameterValue oneStep = (number)1. / (steps - 1);
            ParameterValue numberOfSteps = rnbo_fround(normalizedValue / oneStep * 1 / (number)1) * (number)1;
            normalizedValue = numberOfSteps * oneStep;
        }
    
        return normalizedValue;
    }
    
    ParameterValue convertToNormalizedParameterValue(ParameterIndex index, ParameterValue value) const {
        switch (index) {
        case 0:
            {
            {
                value = (value < 0 ? 0 : (value > 5 ? 5 : value));
                ParameterValue normalizedValue = (value - 0) / (5 - 0);
    
                {
                    normalizedValue = this->applyStepsToNormalizedParameterValue(normalizedValue, 6);
                }
    
                return normalizedValue;
            }
            }
        default:
            {
            return value;
            }
        }
    }
    
    ParameterValue convertFromNormalizedParameterValue(ParameterIndex index, ParameterValue value) const {
        value = (value < 0 ? 0 : (value > 1 ? 1 : value));
    
        switch (index) {
        case 0:
            {
            {
                {
                    value = this->applyStepsToNormalizedParameterValue(value, 6);
                }
    
                {
                    return 0 + value * (5 - 0);
                }
            }
            }
        default:
            {
            return value;
            }
        }
    }
    
    ParameterValue constrainParameterValue(ParameterIndex index, ParameterValue value) const {
        switch (index) {
        case 0:
            {
            return this->param_01_value_constrain(value);
            }
        default:
            {
            return value;
            }
        }
    }
    
    void processNumMessage(MessageTag tag, MessageTag objectId, MillisecondTime time, number payload) {
        this->updateTime(time, (ENGINE*)nullptr);
    
        switch (tag) {
        case TAG("valin"):
            {
            if (TAG("osc.analog/number_obj-39") == objectId)
                this->numberobj_01_valin_set(payload);
    
            break;
            }
        case TAG("format"):
            {
            if (TAG("osc.analog/number_obj-39") == objectId)
                this->numberobj_01_format_set(payload);
    
            break;
            }
        case TAG("sig"):
            {
            if (TAG("osc.analog/number~_obj-18") == objectId)
                this->numbertilde_01_sig_number_set(payload);
    
            if (TAG("osc.analog/number~_obj-25") == objectId)
                this->numbertilde_02_sig_number_set(payload);
    
            break;
            }
        case TAG("mode"):
            {
            if (TAG("osc.analog/number~_obj-18") == objectId)
                this->numbertilde_01_mode_set(payload);
    
            if (TAG("osc.analog/number~_obj-25") == objectId)
                this->numbertilde_02_mode_set(payload);
    
            break;
            }
        }
    }
    
    void processListMessage(
        MessageTag tag,
        MessageTag objectId,
        MillisecondTime time,
        const list& payload
    ) {
        this->updateTime(time, (ENGINE*)nullptr);
    
        switch (tag) {
        case TAG("sig"):
            {
            if (TAG("osc.analog/number~_obj-18") == objectId)
                this->numbertilde_01_sig_list_set(payload);
    
            if (TAG("osc.analog/number~_obj-25") == objectId)
                this->numbertilde_02_sig_list_set(payload);
    
            break;
            }
        }
    }
    
    void processBangMessage(MessageTag , MessageTag , MillisecondTime ) {}
    
    MessageTagInfo resolveTag(MessageTag tag) const {
        switch (tag) {
        case TAG("valout"):
            {
            return "valout";
            }
        case TAG("osc.analog/number_obj-39"):
            {
            return "osc.analog/number_obj-39";
            }
        case TAG("setup"):
            {
            return "setup";
            }
        case TAG("monitor"):
            {
            return "monitor";
            }
        case TAG("osc.analog/number~_obj-18"):
            {
            return "osc.analog/number~_obj-18";
            }
        case TAG("assign"):
            {
            return "assign";
            }
        case TAG("osc.analog/number~_obj-25"):
            {
            return "osc.analog/number~_obj-25";
            }
        case TAG("valin"):
            {
            return "valin";
            }
        case TAG("format"):
            {
            return "format";
            }
        case TAG("sig"):
            {
            return "sig";
            }
        case TAG("mode"):
            {
            return "mode";
            }
        }
    
        return nullptr;
    }
    
    DataRef* getDataRef(DataRefIndex index)  {
        switch (index) {
        default:
            {
            return nullptr;
            }
        }
    }
    
    DataRefIndex getNumDataRefs() const {
        return 0;
    }
    
    void processDataViewUpdate(DataRefIndex index, MillisecondTime time) {
        this->updateTime(time, (ENGINE*)nullptr);
    
        if (index == 0) {
            this->cycle_tilde_01_buffer = reInitDataView(this->cycle_tilde_01_buffer, this->getPatcher()->RNBODefaultSinus);
            this->cycle_tilde_01_bufferUpdated();
        }
    }
    
    void initialize() {
        RNBO_ASSERT(!this->_isInitialized);
        this->assign_defaults();
        this->applyState();
        this->cycle_tilde_01_buffer = new SampleBuffer(this->getPatcher()->RNBODefaultSinus);
        this->_isInitialized = true;
    }
    
    protected:
    
    void updateTime(MillisecondTime time, INTERNALENGINE*, bool inProcess = false) {
    	if (time == TimeNow) time = getTopLevelPatcher()->getPatcherTime();
    	getTopLevelPatcher()->processInternalEvents(time);
    	updateTime(time, (EXTERNALENGINE*)nullptr);
    }
    
    RNBOSubpatcher_23* operator->() {
        return this;
    }
    const RNBOSubpatcher_23* operator->() const {
        return this;
    }
    virtual rnbomatic* getPatcher() const {
        return static_cast<rnbomatic *>(_parentPatcher);
    }
    
    rnbomatic* getTopLevelPatcher() {
        return this->getPatcher()->getTopLevelPatcher();
    }
    
    void cancelClockEvents()
    {
        getEngine()->flushClockEvents(this, 1396722025, false);
        getEngine()->flushClockEvents(this, 694892522, false);
    }
    
    Index voice() {
        return this->_voiceIndex;
    }
    
    number random(number low, number high) {
        number range = high - low;
        return globalrandom() * range + low;
    }
    
    number mstosamps(MillisecondTime ms) {
        return ms * this->sr * 0.001;
    }
    
    number fromnormalized(Index index, number normalizedValue) {
        return this->convertFromNormalizedParameterValue(index, normalizedValue);
    }
    
    void param_01_value_set(number v) {
        v = this->param_01_value_constrain(v);
        this->param_01_value = v;
        this->sendParameter(0, false);
    
        if (this->param_01_value != this->param_01_lastValue) {
            {
                this->getEngine()->presetTouched();
            }
    
            this->param_01_lastValue = this->param_01_value;
        }
    
        this->expr_01_in1_set(v);
        this->numberobj_01_value_set(v);
    }
    
    MillisecondTime getPatcherTime() const {
        return this->_currentTime;
    }
    
    void numberobj_01_valin_set(number v) {
        this->numberobj_01_value_set(v);
    }
    
    void numberobj_01_format_set(number v) {
        this->numberobj_01_currentFormat = trunc((v > 6 ? 6 : (v < 0 ? 0 : v)));
    }
    
    void numbertilde_01_sig_number_set(number v) {
        this->numbertilde_01_outValue = v;
    }
    
    template<typename LISTTYPE> void numbertilde_01_sig_list_set(const LISTTYPE& v) {
        this->numbertilde_01_outValue = v[0];
    }
    
    void numbertilde_01_mode_set(number v) {
        if (v == 1) {
            this->numbertilde_01_currentMode = 0;
        } else if (v == 2) {
            this->numbertilde_01_currentMode = 1;
        }
    }
    
    void numbertilde_02_sig_number_set(number v) {
        this->numbertilde_02_outValue = v;
    }
    
    template<typename LISTTYPE> void numbertilde_02_sig_list_set(const LISTTYPE& v) {
        this->numbertilde_02_outValue = v[0];
    }
    
    void numbertilde_02_mode_set(number v) {
        if (v == 1) {
            this->numbertilde_02_currentMode = 0;
        } else if (v == 2) {
            this->numbertilde_02_currentMode = 1;
        }
    }
    
    void numbertilde_01_value_set(number ) {}
    
    void numbertilde_02_value_set(number ) {}
    
    void deallocateSignals() {
        Index i;
    
        for (i = 0; i < 11; i++) {
            this->signals[i] = freeSignal(this->signals[i]);
        }
    
        this->ip_01_sigbuf = freeSignal(this->ip_01_sigbuf);
        this->zeroBuffer = freeSignal(this->zeroBuffer);
        this->dummyBuffer = freeSignal(this->dummyBuffer);
    }
    
    Index getMaxBlockSize() const {
        return this->maxvs;
    }
    
    number getSampleRate() const {
        return this->sr;
    }
    
    bool hasFixedVectorSize() const {
        return false;
    }
    
    void setProbingTarget(MessageTag ) {}
    
    void initializeObjects() {
        this->numberobj_01_init();
        this->noise_tilde_01_nz_init();
        this->numbertilde_01_init();
        this->numbertilde_02_init();
        this->ip_01_init();
    }
    
    Index getIsMuted()  {
        return this->isMuted;
    }
    
    void setIsMuted(Index v)  {
        this->isMuted = v;
    }
    
    void onSampleRateChanged(double ) {}
    
    void extractState(PatcherStateInterface& ) {}
    
    void applyState() {}
    
    void setParameterOffset(ParameterIndex offset) {
        this->parameterOffset = offset;
    }
    
    void processClockEvent(MillisecondTime time, ClockId index, bool hasValue, ParameterValue value) {
        RNBO_UNUSED(hasValue);
        this->updateTime(time, (ENGINE*)nullptr);
    
        switch (index) {
        case 1396722025:
            {
            this->numbertilde_01_value_set(value);
            break;
            }
        case 694892522:
            {
            this->numbertilde_02_value_set(value);
            break;
            }
        }
    }
    
    void processOutletAtCurrentTime(EngineLink* , OutletIndex , ParameterValue ) {}
    
    void processOutletEvent(
        EngineLink* sender,
        OutletIndex index,
        ParameterValue value,
        MillisecondTime time
    ) {
        this->updateTime(time, (ENGINE*)nullptr);
        this->processOutletAtCurrentTime(sender, index, value);
    }
    
    void sendOutlet(OutletIndex index, ParameterValue value) {
        this->getEngine()->sendOutlet(this, index, value);
    }
    
    void startup() {
        this->updateTime(this->getEngine()->getCurrentTime(), (ENGINE*)nullptr);
    
        {
            this->scheduleParamInit(0, 0);
        }
    }
    
    void fillDataRef(DataRefIndex , DataRef& ) {}
    
    void allocateDataRefs() {
        this->cycle_tilde_01_buffer->requestSize(16384, 1);
        this->cycle_tilde_01_buffer->setSampleRate(this->sr);
        this->cycle_tilde_01_buffer = this->cycle_tilde_01_buffer->allocateIfNeeded();
    }
    
    number param_01_value_constrain(number v) const {
        v = (v > 5 ? 5 : (v < 0 ? 0 : v));
    
        {
            number oneStep = (number)5 / (number)5;
            number oneStepInv = (oneStep != 0 ? (number)1 / oneStep : 0);
            number numberOfSteps = rnbo_fround((v - 0) * oneStepInv * 1 / (number)1) * 1;
            v = numberOfSteps * oneStep + 0;
        }
    
        return v;
    }
    
    void selector_02_onoff_set(number v) {
        this->selector_02_onoff = v;
    }
    
    void selector_01_onoff_set(number v) {
        this->selector_01_onoff = v;
    }
    
    void expr_01_out1_set(number v) {
        this->expr_01_out1 = v;
        this->selector_02_onoff_set(this->expr_01_out1);
        this->selector_01_onoff_set(this->expr_01_out1);
    }
    
    void expr_01_in1_set(number in1) {
        this->expr_01_in1 = in1;
        this->expr_01_out1_set(this->expr_01_in1 + this->expr_01_in2);//#map:osc.analog/+_obj-7:1
    }
    
    void numberobj_01_output_set(number ) {}
    
    void numberobj_01_value_set(number v) {
        this->numberobj_01_value_setter(v);
        v = this->numberobj_01_value;
        number localvalue = v;
    
        if (this->numberobj_01_currentFormat != 6) {
            localvalue = trunc(localvalue);
        }
    
        this->numberobj_01_output_set(localvalue);
    }
    
    void ctlin_01_outchannel_set(number ) {}
    
    void ctlin_01_outcontroller_set(number ) {}
    
    void fromnormalized_01_output_set(number v) {
        this->param_01_value_set(v);
    }
    
    void fromnormalized_01_input_set(number v) {
        this->fromnormalized_01_output_set(this->fromnormalized(0, v));
    }
    
    void expr_02_out1_set(number v) {
        this->expr_02_out1 = v;
        this->fromnormalized_01_input_set(this->expr_02_out1);
    }
    
    void expr_02_in1_set(number in1) {
        this->expr_02_in1 = in1;
        this->expr_02_out1_set(this->expr_02_in1 * this->expr_02_in2);//#map:expr_02:1
    }
    
    void ctlin_01_value_set(number v) {
        this->expr_02_in1_set(v);
    }
    
    void ctlin_01_midihandler(int status, int channel, int port, ConstByteArray data, Index length) {
        RNBO_UNUSED(length);
        RNBO_UNUSED(port);
    
        if (status == 0xB0 && (channel == this->ctlin_01_channel || this->ctlin_01_channel == -1) && (data[1] == this->ctlin_01_controller || this->ctlin_01_controller == -1)) {
            this->ctlin_01_outchannel_set(channel);
            this->ctlin_01_outcontroller_set(data[1]);
            this->ctlin_01_value_set(data[2]);
            this->ctlin_01_status = 0;
        }
    }
    
    void noise_tilde_01_perform(SampleValue * out, Index n) {
        for (Index i = 0; i < n; i++) {
            out[(Index)i] = this->noise_tilde_01_nz_next();
        }
    }
    
    void cycle_tilde_01_perform(
        const Sample * frequency,
        number phase_offset,
        SampleValue * out1,
        SampleValue * out2,
        Index n
    ) {
        RNBO_UNUSED(phase_offset);
        auto __cycle_tilde_01_f2i = this->cycle_tilde_01_f2i;
        auto __cycle_tilde_01_buffer = this->cycle_tilde_01_buffer;
        auto __cycle_tilde_01_phasei = this->cycle_tilde_01_phasei;
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            {
                UInt32 uint_phase;
    
                {
                    {
                        uint_phase = __cycle_tilde_01_phasei;
                    }
                }
    
                UInt32 idx = (UInt32)(uint32_rshift(uint_phase, 18));
                number frac = ((BinOpInt)((BinOpInt)uint_phase & (BinOpInt)262143)) * 3.81471181759574e-6;
                number y0 = __cycle_tilde_01_buffer[(Index)idx];
                number y1 = __cycle_tilde_01_buffer[(Index)((BinOpInt)(idx + 1) & (BinOpInt)16383)];
                number y = y0 + frac * (y1 - y0);
    
                {
                    UInt32 pincr = (UInt32)(uint32_trunc(frequency[(Index)i] * __cycle_tilde_01_f2i));
                    __cycle_tilde_01_phasei = uint32_add(__cycle_tilde_01_phasei, pincr);
                }
    
                out1[(Index)i] = y;
                out2[(Index)i] = uint_phase * 0.232830643653869629e-9;
                continue;
            }
        }
    
        this->cycle_tilde_01_phasei = __cycle_tilde_01_phasei;
    }
    
    void numbertilde_01_perform(const SampleValue * input_signal, SampleValue * output, Index n) {
        auto __numbertilde_01_currentIntervalInSamples = this->numbertilde_01_currentIntervalInSamples;
        auto __numbertilde_01_lastValue = this->numbertilde_01_lastValue;
        auto __numbertilde_01_currentInterval = this->numbertilde_01_currentInterval;
        auto __numbertilde_01_rampInSamples = this->numbertilde_01_rampInSamples;
        auto __numbertilde_01_outValue = this->numbertilde_01_outValue;
        auto __numbertilde_01_currentMode = this->numbertilde_01_currentMode;
        number monitorvalue = input_signal[0];
    
        for (Index i = 0; i < n; i++) {
            if (__numbertilde_01_currentMode == 0) {
                output[(Index)i] = this->numbertilde_01_smooth_next(
                    __numbertilde_01_outValue,
                    __numbertilde_01_rampInSamples,
                    __numbertilde_01_rampInSamples
                );
            } else {
                output[(Index)i] = input_signal[(Index)i];
            }
        }
    
        __numbertilde_01_currentInterval -= n;
    
        if (monitorvalue != __numbertilde_01_lastValue && __numbertilde_01_currentInterval <= 0) {
            __numbertilde_01_currentInterval = __numbertilde_01_currentIntervalInSamples;
    
            this->getEngine()->scheduleClockEventWithValue(
                this,
                1396722025,
                this->sampsToMs((SampleIndex)(this->vs)) + this->_currentTime,
                monitorvalue
            );;
    
            __numbertilde_01_lastValue = monitorvalue;
            ;
        }
    
        this->numbertilde_01_currentInterval = __numbertilde_01_currentInterval;
        this->numbertilde_01_lastValue = __numbertilde_01_lastValue;
    }
    
    void saw_tilde_01_perform(
        const Sample * frequency,
        number syncPhase,
        SampleValue * out1,
        SampleValue * out2,
        Index n
    ) {
        RNBO_UNUSED(syncPhase);
        auto __saw_tilde_01_didSync = this->saw_tilde_01_didSync;
        auto __saw_tilde_01_lastSyncDiff = this->saw_tilde_01_lastSyncDiff;
        auto __saw_tilde_01_lastSyncPhase = this->saw_tilde_01_lastSyncPhase;
        auto __saw_tilde_01_t = this->saw_tilde_01_t;
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            number dt = frequency[(Index)i] / this->sr;
            number t1 = __saw_tilde_01_t + 0.5;
            t1 -= trunc(t1);
            number y = 2 * t1 - 1;
    
            if (dt != 0.0) {
                number syncDiff = 0 - __saw_tilde_01_lastSyncPhase;
                __saw_tilde_01_lastSyncPhase = 0;
                __saw_tilde_01_lastSyncDiff = syncDiff;
                number lookahead = 0 + syncDiff;
    
                if (t1 < dt) {
                    number d = t1 / dt;
                    y -= d + d - d * d - 1;
                } else if (t1 + dt > 1) {
                    number d = (t1 - 1) / dt;
                    y -= d + d + d * d + 1;
                } else if ((bool)(__saw_tilde_01_didSync)) {
                    y = 0;
                    __saw_tilde_01_didSync = false;
                } else if (lookahead > 1) {
                    y *= 0.5;
                    __saw_tilde_01_t = 0;
                    __saw_tilde_01_didSync = true;
                }
    
                __saw_tilde_01_t += dt;
    
                if (dt > 0) {
                    while (__saw_tilde_01_t >= 1) {
                        __saw_tilde_01_t -= 1;
                    }
                } else {
                    while (__saw_tilde_01_t <= 0) {
                        __saw_tilde_01_t += 1;
                    }
                }
            }
    
            y = this->saw_tilde_01_dcblocker_next(y, 0.9997);
            out1[(Index)i] = 0.5 * y;
            out2[(Index)i] = __saw_tilde_01_t;
        }
    
        this->saw_tilde_01_t = __saw_tilde_01_t;
        this->saw_tilde_01_lastSyncPhase = __saw_tilde_01_lastSyncPhase;
        this->saw_tilde_01_lastSyncDiff = __saw_tilde_01_lastSyncDiff;
        this->saw_tilde_01_didSync = __saw_tilde_01_didSync;
    }
    
    void rect_tilde_01_perform(
        const Sample * frequency,
        const Sample * pulsewidth,
        number syncPhase,
        SampleValue * out1,
        SampleValue * out2,
        Index n
    ) {
        RNBO_UNUSED(syncPhase);
        auto __rect_tilde_01_xHistory = this->rect_tilde_01_xHistory;
        auto __rect_tilde_01_yHistory = this->rect_tilde_01_yHistory;
        auto __rect_tilde_01_didSync = this->rect_tilde_01_didSync;
        auto __rect_tilde_01_t = this->rect_tilde_01_t;
        auto __rect_tilde_01_lastSyncDiff = this->rect_tilde_01_lastSyncDiff;
        auto __rect_tilde_01_lastSyncPhase = this->rect_tilde_01_lastSyncPhase;
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            number __frequency = frequency[(Index)i];
            __frequency = rnbo_abs(__frequency);
            number dt = __frequency / this->sr;
            number pw = pulsewidth[(Index)i];
    
            if (pulsewidth[(Index)i] > 0.99) {
                pw = 0.99;
            } else if (pulsewidth[(Index)i] < 0.01) {
                pw = 0.01;
            }
    
            number syncDiff = 0 - __rect_tilde_01_lastSyncPhase;
            __rect_tilde_01_lastSyncPhase = 0;
            __rect_tilde_01_lastSyncDiff = syncDiff;
            number syncLookahead = 0 + syncDiff;
            number tCurr = __rect_tilde_01_t;
            number tPrev = tCurr - dt;
            number tNext = tCurr + dt;
    
            if (tPrev < 0) {
                while (tPrev < 0) {
                    tPrev += 1;
                }
            }
    
            if (tNext > 1) {
                while (tNext >= 1) {
                    tNext -= 1;
                }
            }
    
            number yNext = this->rect_tilde_01_rectangle(tNext, pw);
            number yCurr = this->rect_tilde_01_rectangle(tCurr, pw);
            number yPrev = this->rect_tilde_01_rectangle(tPrev, pw);
    
            if (dt != 0.0) {
                if (yPrev < yCurr) {
                    number d = tCurr / dt;
                    yCurr += d - 0.5 * d * d - 0.5;
                } else if (yCurr < yNext) {
                    number d = (1 - tCurr) / dt;
                    yCurr += 0.5 * d * d + d + 0.5;
                } else if (yPrev > yCurr) {
                    number d = (tCurr - pw) / dt;
                    yCurr -= d - 0.5 * d * d - 0.5;
                } else if (yCurr > yNext) {
                    number d = (pw - tCurr) / dt;
                    yCurr -= 0.5 * d * d + d + 0.5;
                } else if ((bool)(__rect_tilde_01_didSync)) {
                    yCurr = 0.25;
                    __rect_tilde_01_didSync = false;
                } else if (syncLookahead > 1) {
                    if (yCurr < 0) {
                        yCurr = -0.125;
                    }
    
                    __rect_tilde_01_t = 0;
                    __rect_tilde_01_didSync = true;
                }
    
                __rect_tilde_01_t += dt;
    
                if (dt > 0) {
                    while (__rect_tilde_01_t >= 1) {
                        __rect_tilde_01_t -= 1;
                    }
                } else {
                    while (__rect_tilde_01_t <= 0) {
                        __rect_tilde_01_t += 1;
                    }
                }
            }
    
            number output = yCurr - __rect_tilde_01_yHistory + __rect_tilde_01_xHistory * 0.9997;
            __rect_tilde_01_xHistory = output;
            __rect_tilde_01_yHistory = yCurr;
            out1[(Index)i] = 0.5 * output;
            out2[(Index)i] = __rect_tilde_01_t;
        }
    
        this->rect_tilde_01_lastSyncPhase = __rect_tilde_01_lastSyncPhase;
        this->rect_tilde_01_lastSyncDiff = __rect_tilde_01_lastSyncDiff;
        this->rect_tilde_01_t = __rect_tilde_01_t;
        this->rect_tilde_01_didSync = __rect_tilde_01_didSync;
        this->rect_tilde_01_yHistory = __rect_tilde_01_yHistory;
        this->rect_tilde_01_xHistory = __rect_tilde_01_xHistory;
    }
    
    void tri_tilde_01_perform(
        const Sample * frequency,
        const Sample * pulsewidth,
        number syncPhase,
        SampleValue * out1,
        SampleValue * out2,
        Index n
    ) {
        RNBO_UNUSED(syncPhase);
        auto __tri_tilde_01_yn3 = this->tri_tilde_01_yn3;
        auto __tri_tilde_01_yn2 = this->tri_tilde_01_yn2;
        auto __tri_tilde_01_yn1 = this->tri_tilde_01_yn1;
        auto __tri_tilde_01_app_correction = this->tri_tilde_01_app_correction;
        auto __tri_tilde_01_flg = this->tri_tilde_01_flg;
        auto __tri_tilde_01_yn = this->tri_tilde_01_yn;
        auto __tri_tilde_01_t = this->tri_tilde_01_t;
        auto __tri_tilde_01_lastSyncDiff = this->tri_tilde_01_lastSyncDiff;
        auto __tri_tilde_01_lastSyncPhase = this->tri_tilde_01_lastSyncPhase;
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            number __frequency = frequency[(Index)i];
            __frequency = rnbo_abs(__frequency);
            number dt = __frequency / this->sr;
    
            if (dt != 0.0) {
                number pw = pulsewidth[(Index)i];
    
                if (pulsewidth[(Index)i] > 0.99) {
                    pw = 0.99;
                } else if (pulsewidth[(Index)i] < 0.01) {
                    pw = 0.01;
                }
    
                number syncDiff = 0 - __tri_tilde_01_lastSyncPhase;
                __tri_tilde_01_lastSyncPhase = 0;
                __tri_tilde_01_lastSyncDiff = syncDiff;
                number syncLookahead = 0 + syncDiff;
    
                if (syncLookahead > 1) {
                    __tri_tilde_01_t = 0;
                }
    
                number tCurr = __tri_tilde_01_t;
                number upSlope = __frequency / (pw * this->sr);
                number downSlope = __frequency / ((1 - pw) * this->sr);
    
                if (tCurr <= pw) {
                    __tri_tilde_01_yn = (number)2 / pw * tCurr - 1;
    
                    if (__tri_tilde_01_flg == -1) {
                        __tri_tilde_01_app_correction = 1;
                        __tri_tilde_01_flg = 1;
                    } else if (__tri_tilde_01_app_correction == 1) {
                        __tri_tilde_01_app_correction = 0;
                        number d = (tCurr - dt) / dt;
                        number d2 = d * d;
                        number d3 = d2 * d;
                        number d4 = d2 * d2;
                        number d5 = d * d4;
                        number h0 = -d5 / (number)120 + d4 / (number)24 - d3 / (number)12 + d2 / (number)12 - d / (number)24 + (number)1 / (number)120;
                        number h1 = d5 / (number)40 - d4 / (number)12 + d2 / (number)3 - d / (number)2 + (number)7 / (number)30;
                        number h2 = -d5 / (number)40 + d4 / (number)24 + d3 / (number)12 + d2 / (number)12 + d / (number)24 + (number)1 / (number)120;
                        number h3 = d5 / (number)120;
                        __tri_tilde_01_yn += upSlope * h0;
                        __tri_tilde_01_yn1 += upSlope * h1;
                        __tri_tilde_01_yn2 += upSlope * h2;
                        __tri_tilde_01_yn3 += upSlope * h3;
                    }
    
                    __tri_tilde_01_flg = 1;
                } else {
                    __tri_tilde_01_yn = 1 - 2 * (tCurr - pw) / (1 - pw);
    
                    if (__tri_tilde_01_flg == 1) {
                        __tri_tilde_01_app_correction = 1;
                    } else if (__tri_tilde_01_app_correction == 1) {
                        __tri_tilde_01_app_correction = 0;
                        number d = (tCurr - pw - dt) / dt;
                        number d2 = d * d;
                        number d3 = d2 * d;
                        number d4 = d2 * d2;
                        number d5 = d4 * d;
                        number h0 = -d5 / (number)120 + d4 / (number)24 - d3 / (number)12 + d2 / (number)12 - d / (number)24 + (number)1 / (number)120;
                        number h1 = d5 / (number)40 - d4 / (number)12 + d2 / (number)3 - d / (number)2 + (number)7 / (number)30;
                        number h2 = -d5 / (number)40 + d4 / (number)24 + d3 / (number)12 + d2 / (number)12 + d / (number)24 + (number)1 / (number)120;
                        number h3 = d5 / (number)120;
                        __tri_tilde_01_yn -= downSlope * h0;
                        __tri_tilde_01_yn1 -= downSlope * h1;
                        __tri_tilde_01_yn2 -= downSlope * h2;
                        __tri_tilde_01_yn3 -= downSlope * h3;
                    }
    
                    __tri_tilde_01_flg = -1;
                }
            }
    
            number y = __tri_tilde_01_yn3;
            __tri_tilde_01_yn3 = __tri_tilde_01_yn2;
            __tri_tilde_01_yn2 = __tri_tilde_01_yn1;
            __tri_tilde_01_yn1 = __tri_tilde_01_yn;
            __tri_tilde_01_t += dt;
    
            if (dt > 0) {
                while (__tri_tilde_01_t >= 1) {
                    __tri_tilde_01_t -= 1;
                }
            } else {
                while (__tri_tilde_01_t <= 0) {
                    __tri_tilde_01_t += 1;
                }
            }
    
            y = this->tri_tilde_01_dcblocker_next(y, 0.9997);
            out1[(Index)i] = y * 0.5;
            out2[(Index)i] = __tri_tilde_01_t;
        }
    
        this->tri_tilde_01_lastSyncPhase = __tri_tilde_01_lastSyncPhase;
        this->tri_tilde_01_lastSyncDiff = __tri_tilde_01_lastSyncDiff;
        this->tri_tilde_01_t = __tri_tilde_01_t;
        this->tri_tilde_01_yn = __tri_tilde_01_yn;
        this->tri_tilde_01_flg = __tri_tilde_01_flg;
        this->tri_tilde_01_app_correction = __tri_tilde_01_app_correction;
        this->tri_tilde_01_yn1 = __tri_tilde_01_yn1;
        this->tri_tilde_01_yn2 = __tri_tilde_01_yn2;
        this->tri_tilde_01_yn3 = __tri_tilde_01_yn3;
    }
    
    void numbertilde_02_perform(const SampleValue * input_signal, SampleValue * output, Index n) {
        auto __numbertilde_02_currentIntervalInSamples = this->numbertilde_02_currentIntervalInSamples;
        auto __numbertilde_02_lastValue = this->numbertilde_02_lastValue;
        auto __numbertilde_02_currentInterval = this->numbertilde_02_currentInterval;
        auto __numbertilde_02_rampInSamples = this->numbertilde_02_rampInSamples;
        auto __numbertilde_02_outValue = this->numbertilde_02_outValue;
        auto __numbertilde_02_currentMode = this->numbertilde_02_currentMode;
        number monitorvalue = input_signal[0];
    
        for (Index i = 0; i < n; i++) {
            if (__numbertilde_02_currentMode == 0) {
                output[(Index)i] = this->numbertilde_02_smooth_next(
                    __numbertilde_02_outValue,
                    __numbertilde_02_rampInSamples,
                    __numbertilde_02_rampInSamples
                );
            } else {
                output[(Index)i] = input_signal[(Index)i];
            }
        }
    
        __numbertilde_02_currentInterval -= n;
    
        if (monitorvalue != __numbertilde_02_lastValue && __numbertilde_02_currentInterval <= 0) {
            __numbertilde_02_currentInterval = __numbertilde_02_currentIntervalInSamples;
    
            this->getEngine()->scheduleClockEventWithValue(
                this,
                694892522,
                this->sampsToMs((SampleIndex)(this->vs)) + this->_currentTime,
                monitorvalue
            );;
    
            __numbertilde_02_lastValue = monitorvalue;
            ;
        }
    
        this->numbertilde_02_currentInterval = __numbertilde_02_currentInterval;
        this->numbertilde_02_lastValue = __numbertilde_02_lastValue;
    }
    
    void rect_tilde_02_perform(
        const Sample * frequency,
        const Sample * pulsewidth,
        number syncPhase,
        SampleValue * out1,
        SampleValue * out2,
        Index n
    ) {
        RNBO_UNUSED(syncPhase);
        auto __rect_tilde_02_xHistory = this->rect_tilde_02_xHistory;
        auto __rect_tilde_02_yHistory = this->rect_tilde_02_yHistory;
        auto __rect_tilde_02_didSync = this->rect_tilde_02_didSync;
        auto __rect_tilde_02_t = this->rect_tilde_02_t;
        auto __rect_tilde_02_lastSyncDiff = this->rect_tilde_02_lastSyncDiff;
        auto __rect_tilde_02_lastSyncPhase = this->rect_tilde_02_lastSyncPhase;
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            number __frequency = frequency[(Index)i];
            __frequency = rnbo_abs(__frequency);
            number dt = __frequency / this->sr;
            number pw = pulsewidth[(Index)i];
    
            if (pulsewidth[(Index)i] > 0.99) {
                pw = 0.99;
            } else if (pulsewidth[(Index)i] < 0.01) {
                pw = 0.01;
            }
    
            number syncDiff = 0 - __rect_tilde_02_lastSyncPhase;
            __rect_tilde_02_lastSyncPhase = 0;
            __rect_tilde_02_lastSyncDiff = syncDiff;
            number syncLookahead = 0 + syncDiff;
            number tCurr = __rect_tilde_02_t;
            number tPrev = tCurr - dt;
            number tNext = tCurr + dt;
    
            if (tPrev < 0) {
                while (tPrev < 0) {
                    tPrev += 1;
                }
            }
    
            if (tNext > 1) {
                while (tNext >= 1) {
                    tNext -= 1;
                }
            }
    
            number yNext = this->rect_tilde_02_rectangle(tNext, pw);
            number yCurr = this->rect_tilde_02_rectangle(tCurr, pw);
            number yPrev = this->rect_tilde_02_rectangle(tPrev, pw);
    
            if (dt != 0.0) {
                if (yPrev < yCurr) {
                    number d = tCurr / dt;
                    yCurr += d - 0.5 * d * d - 0.5;
                } else if (yCurr < yNext) {
                    number d = (1 - tCurr) / dt;
                    yCurr += 0.5 * d * d + d + 0.5;
                } else if (yPrev > yCurr) {
                    number d = (tCurr - pw) / dt;
                    yCurr -= d - 0.5 * d * d - 0.5;
                } else if (yCurr > yNext) {
                    number d = (pw - tCurr) / dt;
                    yCurr -= 0.5 * d * d + d + 0.5;
                } else if ((bool)(__rect_tilde_02_didSync)) {
                    yCurr = 0.25;
                    __rect_tilde_02_didSync = false;
                } else if (syncLookahead > 1) {
                    if (yCurr < 0) {
                        yCurr = -0.125;
                    }
    
                    __rect_tilde_02_t = 0;
                    __rect_tilde_02_didSync = true;
                }
    
                __rect_tilde_02_t += dt;
    
                if (dt > 0) {
                    while (__rect_tilde_02_t >= 1) {
                        __rect_tilde_02_t -= 1;
                    }
                } else {
                    while (__rect_tilde_02_t <= 0) {
                        __rect_tilde_02_t += 1;
                    }
                }
            }
    
            number output = yCurr - __rect_tilde_02_yHistory + __rect_tilde_02_xHistory * 0.9997;
            __rect_tilde_02_xHistory = output;
            __rect_tilde_02_yHistory = yCurr;
            out1[(Index)i] = 0.5 * output;
            out2[(Index)i] = __rect_tilde_02_t;
        }
    
        this->rect_tilde_02_lastSyncPhase = __rect_tilde_02_lastSyncPhase;
        this->rect_tilde_02_lastSyncDiff = __rect_tilde_02_lastSyncDiff;
        this->rect_tilde_02_t = __rect_tilde_02_t;
        this->rect_tilde_02_didSync = __rect_tilde_02_didSync;
        this->rect_tilde_02_yHistory = __rect_tilde_02_yHistory;
        this->rect_tilde_02_xHistory = __rect_tilde_02_xHistory;
    }
    
    void selector_01_perform(
        number onoff,
        const SampleValue * in1,
        const SampleValue * in2,
        const SampleValue * in3,
        const SampleValue * in4,
        const SampleValue * in5,
        const SampleValue * in6,
        SampleValue * out,
        Index n
    ) {
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            if (onoff >= 1 && onoff < 2)
                out[(Index)i] = in1[(Index)i];
            else if (onoff >= 2 && onoff < 3)
                out[(Index)i] = in2[(Index)i];
            else if (onoff >= 3 && onoff < 4)
                out[(Index)i] = in3[(Index)i];
            else if (onoff >= 4 && onoff < 5)
                out[(Index)i] = in4[(Index)i];
            else if (onoff >= 5 && onoff < 6)
                out[(Index)i] = in5[(Index)i];
            else if (onoff >= 6 && onoff < 7)
                out[(Index)i] = in6[(Index)i];
            else
                out[(Index)i] = 0;
        }
    }
    
    void ip_01_perform(SampleValue * out, Index n) {
        auto __ip_01_lastValue = this->ip_01_lastValue;
        auto __ip_01_lastIndex = this->ip_01_lastIndex;
    
        for (Index i = 0; i < n; i++) {
            out[(Index)i] = ((SampleIndex)(i) >= __ip_01_lastIndex ? __ip_01_lastValue : this->ip_01_sigbuf[(Index)i]);
        }
    
        __ip_01_lastIndex = 0;
        this->ip_01_lastIndex = __ip_01_lastIndex;
    }
    
    void selector_02_perform(
        number onoff,
        const SampleValue * in1,
        const SampleValue * in2,
        const SampleValue * in3,
        const SampleValue * in4,
        const SampleValue * in5,
        const SampleValue * in6,
        SampleValue * out,
        Index n
    ) {
        Index i;
    
        for (i = 0; i < (Index)n; i++) {
            if (onoff >= 1 && onoff < 2)
                out[(Index)i] = in1[(Index)i];
            else if (onoff >= 2 && onoff < 3)
                out[(Index)i] = in2[(Index)i];
            else if (onoff >= 3 && onoff < 4)
                out[(Index)i] = in3[(Index)i];
            else if (onoff >= 4 && onoff < 5)
                out[(Index)i] = in4[(Index)i];
            else if (onoff >= 5 && onoff < 6)
                out[(Index)i] = in5[(Index)i];
            else if (onoff >= 6 && onoff < 7)
                out[(Index)i] = in6[(Index)i];
            else
                out[(Index)i] = 0;
        }
    }
    
    void stackprotect_perform(Index n) {
        RNBO_UNUSED(n);
        auto __stackprotect_count = this->stackprotect_count;
        __stackprotect_count = 0;
        this->stackprotect_count = __stackprotect_count;
    }
    
    void numberobj_01_value_setter(number v) {
        number localvalue = v;
    
        if (this->numberobj_01_currentFormat != 6) {
            localvalue = trunc(localvalue);
        }
    
        this->numberobj_01_value = localvalue;
    }
    
    void numberobj_01_init() {
        this->numberobj_01_currentFormat = 6;
    }
    
    void numberobj_01_getPresetValue(PatcherStateInterface& preset) {
        preset["value"] = this->numberobj_01_value;
    }
    
    void numberobj_01_setPresetValue(PatcherStateInterface& preset) {
        if ((bool)(stateIsEmpty(preset)))
            return;
    
        this->numberobj_01_value_set(preset["value"]);
    }
    
    void param_01_getPresetValue(PatcherStateInterface& preset) {
        preset["value"] = this->param_01_value;
    }
    
    void param_01_setPresetValue(PatcherStateInterface& preset) {
        if ((bool)(stateIsEmpty(preset)))
            return;
    
        this->param_01_value_set(preset["value"]);
    }
    
    void noise_tilde_01_nz_reset() {
        xoshiro_reset(
            systemticks() + this->voice() + this->random(0, 10000),
            this->noise_tilde_01_nz_state
        );
    }
    
    void noise_tilde_01_nz_init() {
        this->noise_tilde_01_nz_reset();
    }
    
    void noise_tilde_01_nz_seed(number v) {
        xoshiro_reset(v, this->noise_tilde_01_nz_state);
    }
    
    number noise_tilde_01_nz_next() {
        return xoshiro_next(this->noise_tilde_01_nz_state);
    }
    
    void noise_tilde_01_dspsetup(bool force) {
        if ((bool)(this->noise_tilde_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->noise_tilde_01_setupDone = true;
    }
    
    number cycle_tilde_01_ph_next(number freq, number reset) {
        {
            {
                if (reset >= 0.)
                    this->cycle_tilde_01_ph_currentPhase = reset;
            }
        }
    
        number pincr = freq * this->cycle_tilde_01_ph_conv;
    
        if (this->cycle_tilde_01_ph_currentPhase < 0.)
            this->cycle_tilde_01_ph_currentPhase = 1. + this->cycle_tilde_01_ph_currentPhase;
    
        if (this->cycle_tilde_01_ph_currentPhase > 1.)
            this->cycle_tilde_01_ph_currentPhase = this->cycle_tilde_01_ph_currentPhase - 1.;
    
        number tmp = this->cycle_tilde_01_ph_currentPhase;
        this->cycle_tilde_01_ph_currentPhase += pincr;
        return tmp;
    }
    
    void cycle_tilde_01_ph_reset() {
        this->cycle_tilde_01_ph_currentPhase = 0;
    }
    
    void cycle_tilde_01_ph_dspsetup() {
        this->cycle_tilde_01_ph_conv = (number)1 / this->sr;
    }
    
    void cycle_tilde_01_dspsetup(bool force) {
        if ((bool)(this->cycle_tilde_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->cycle_tilde_01_phasei = 0;
        this->cycle_tilde_01_f2i = (number)4294967296 / this->sr;
        this->cycle_tilde_01_wrap = (Int)(this->cycle_tilde_01_buffer->getSize()) - 1;
        this->cycle_tilde_01_setupDone = true;
        this->cycle_tilde_01_ph_dspsetup();
    }
    
    void cycle_tilde_01_bufferUpdated() {
        this->cycle_tilde_01_wrap = (Int)(this->cycle_tilde_01_buffer->getSize()) - 1;
    }
    
    number numbertilde_01_smooth_d_next(number x) {
        number temp = (number)(x - this->numbertilde_01_smooth_d_prev);
        this->numbertilde_01_smooth_d_prev = x;
        return temp;
    }
    
    void numbertilde_01_smooth_d_dspsetup() {
        this->numbertilde_01_smooth_d_reset();
    }
    
    void numbertilde_01_smooth_d_reset() {
        this->numbertilde_01_smooth_d_prev = 0;
    }
    
    number numbertilde_01_smooth_next(number x, number up, number down) {
        if (this->numbertilde_01_smooth_d_next(x) != 0.) {
            if (x > this->numbertilde_01_smooth_prev) {
                number _up = up;
    
                if (_up < 1)
                    _up = 1;
    
                this->numbertilde_01_smooth_index = _up;
                this->numbertilde_01_smooth_increment = (x - this->numbertilde_01_smooth_prev) / _up;
            } else if (x < this->numbertilde_01_smooth_prev) {
                number _down = down;
    
                if (_down < 1)
                    _down = 1;
    
                this->numbertilde_01_smooth_index = _down;
                this->numbertilde_01_smooth_increment = (x - this->numbertilde_01_smooth_prev) / _down;
            }
        }
    
        if (this->numbertilde_01_smooth_index > 0) {
            this->numbertilde_01_smooth_prev += this->numbertilde_01_smooth_increment;
            this->numbertilde_01_smooth_index -= 1;
        } else {
            this->numbertilde_01_smooth_prev = x;
        }
    
        return this->numbertilde_01_smooth_prev;
    }
    
    void numbertilde_01_smooth_reset() {
        this->numbertilde_01_smooth_prev = 0;
        this->numbertilde_01_smooth_index = 0;
        this->numbertilde_01_smooth_increment = 0;
        this->numbertilde_01_smooth_d_reset();
    }
    
    void numbertilde_01_init() {
        this->numbertilde_01_currentMode = 1;
    }
    
    void numbertilde_01_dspsetup(bool force) {
        if ((bool)(this->numbertilde_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->numbertilde_01_currentIntervalInSamples = this->mstosamps(100);
        this->numbertilde_01_currentInterval = this->numbertilde_01_currentIntervalInSamples;
        this->numbertilde_01_rampInSamples = this->mstosamps(this->numbertilde_01_ramp);
        this->numbertilde_01_setupDone = true;
        this->numbertilde_01_smooth_d_dspsetup();
    }
    
    number saw_tilde_01_dcblocker_next(number x, number gain) {
        number y = x - this->saw_tilde_01_dcblocker_xm1 + this->saw_tilde_01_dcblocker_ym1 * gain;
        this->saw_tilde_01_dcblocker_xm1 = x;
        this->saw_tilde_01_dcblocker_ym1 = y;
        return y;
    }
    
    void saw_tilde_01_dcblocker_reset() {
        this->saw_tilde_01_dcblocker_xm1 = 0;
        this->saw_tilde_01_dcblocker_ym1 = 0;
    }
    
    void saw_tilde_01_dcblocker_dspsetup() {
        this->saw_tilde_01_dcblocker_reset();
    }
    
    void saw_tilde_01_dspsetup(bool force) {
        if ((bool)(this->saw_tilde_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->saw_tilde_01_setupDone = true;
        this->saw_tilde_01_dcblocker_dspsetup();
    }
    
    number numbertilde_02_smooth_d_next(number x) {
        number temp = (number)(x - this->numbertilde_02_smooth_d_prev);
        this->numbertilde_02_smooth_d_prev = x;
        return temp;
    }
    
    void numbertilde_02_smooth_d_dspsetup() {
        this->numbertilde_02_smooth_d_reset();
    }
    
    void numbertilde_02_smooth_d_reset() {
        this->numbertilde_02_smooth_d_prev = 0;
    }
    
    number numbertilde_02_smooth_next(number x, number up, number down) {
        if (this->numbertilde_02_smooth_d_next(x) != 0.) {
            if (x > this->numbertilde_02_smooth_prev) {
                number _up = up;
    
                if (_up < 1)
                    _up = 1;
    
                this->numbertilde_02_smooth_index = _up;
                this->numbertilde_02_smooth_increment = (x - this->numbertilde_02_smooth_prev) / _up;
            } else if (x < this->numbertilde_02_smooth_prev) {
                number _down = down;
    
                if (_down < 1)
                    _down = 1;
    
                this->numbertilde_02_smooth_index = _down;
                this->numbertilde_02_smooth_increment = (x - this->numbertilde_02_smooth_prev) / _down;
            }
        }
    
        if (this->numbertilde_02_smooth_index > 0) {
            this->numbertilde_02_smooth_prev += this->numbertilde_02_smooth_increment;
            this->numbertilde_02_smooth_index -= 1;
        } else {
            this->numbertilde_02_smooth_prev = x;
        }
    
        return this->numbertilde_02_smooth_prev;
    }
    
    void numbertilde_02_smooth_reset() {
        this->numbertilde_02_smooth_prev = 0;
        this->numbertilde_02_smooth_index = 0;
        this->numbertilde_02_smooth_increment = 0;
        this->numbertilde_02_smooth_d_reset();
    }
    
    void numbertilde_02_init() {
        this->numbertilde_02_currentMode = 1;
    }
    
    void numbertilde_02_dspsetup(bool force) {
        if ((bool)(this->numbertilde_02_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->numbertilde_02_currentIntervalInSamples = this->mstosamps(100);
        this->numbertilde_02_currentInterval = this->numbertilde_02_currentIntervalInSamples;
        this->numbertilde_02_rampInSamples = this->mstosamps(this->numbertilde_02_ramp);
        this->numbertilde_02_setupDone = true;
        this->numbertilde_02_smooth_d_dspsetup();
    }
    
    number tri_tilde_01_dcblocker_next(number x, number gain) {
        number y = x - this->tri_tilde_01_dcblocker_xm1 + this->tri_tilde_01_dcblocker_ym1 * gain;
        this->tri_tilde_01_dcblocker_xm1 = x;
        this->tri_tilde_01_dcblocker_ym1 = y;
        return y;
    }
    
    void tri_tilde_01_dcblocker_reset() {
        this->tri_tilde_01_dcblocker_xm1 = 0;
        this->tri_tilde_01_dcblocker_ym1 = 0;
    }
    
    void tri_tilde_01_dcblocker_dspsetup() {
        this->tri_tilde_01_dcblocker_reset();
    }
    
    void tri_tilde_01_dspsetup(bool force) {
        if ((bool)(this->tri_tilde_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->tri_tilde_01_setupDone = true;
        this->tri_tilde_01_dcblocker_dspsetup();
    }
    
    void ip_01_init() {
        this->ip_01_lastValue = this->ip_01_value;
    }
    
    void ip_01_dspsetup(bool force) {
        if ((bool)(this->ip_01_setupDone) && (bool)(!(bool)(force)))
            return;
    
        this->ip_01_lastIndex = 0;
        this->ip_01_setupDone = true;
    }
    
    number rect_tilde_01_rectangle(number phase, number pulsewidth) {
        if (phase < pulsewidth) {
            return 1;
        } else {
            return -1;
        }
    }
    
    number rect_tilde_02_rectangle(number phase, number pulsewidth) {
        if (phase < pulsewidth) {
            return 1;
        } else {
            return -1;
        }
    }
    
    bool stackprotect_check() {
        this->stackprotect_count++;
    
        if (this->stackprotect_count > 128) {
            console->log("STACK OVERFLOW DETECTED - stopped processing branch !");
            return true;
        }
    
        return false;
    }
    
    Index getPatcherSerial() const {
        return 0;
    }
    
    void sendParameter(ParameterIndex index, bool ignoreValue) {
        this->getPatcher()->sendParameter(index + this->parameterOffset, ignoreValue);
    }
    
    void scheduleParamInit(ParameterIndex index, Index order) {
        this->getPatcher()->scheduleParamInit(index + this->parameterOffset, order);
    }
    
    void updateTime(MillisecondTime time, EXTERNALENGINE* engine, bool inProcess = false) {
        RNBO_UNUSED(inProcess);
        RNBO_UNUSED(engine);
        this->_currentTime = time;
        auto offset = rnbo_fround(this->msToSamps(time - this->getEngine()->getCurrentTime(), this->sr));
    
        if (offset >= (SampleIndex)(this->vs))
            offset = (SampleIndex)(this->vs) - 1;
    
        if (offset < 0)
            offset = 0;
    
        this->sampleOffsetIntoNextAudioBuffer = (Index)(offset);
    }
    
    void assign_defaults()
    {
        numberobj_01_value = 0;
        numberobj_01_value_setter(numberobj_01_value);
        expr_01_in1 = 0;
        expr_01_in2 = 1;
        expr_01_out1 = 0;
        selector_01_onoff = 1;
        param_01_value = 2;
        noise_tilde_01_seed = 0;
        cycle_tilde_01_frequency = 0;
        cycle_tilde_01_phase_offset = 0;
        numbertilde_01_input_number = 0;
        numbertilde_01_ramp = 0;
        saw_tilde_01_frequency = 0;
        saw_tilde_01_syncPhase = 0;
        numbertilde_02_input_number = 0;
        numbertilde_02_ramp = 0;
        tri_tilde_01_frequency = 0;
        tri_tilde_01_pulsewidth = 0.5;
        tri_tilde_01_syncPhase = 0;
        selector_02_onoff = 1;
        ip_01_value = 0;
        ip_01_impulse = 0;
        rect_tilde_01_frequency = 440;
        rect_tilde_01_pulsewidth = 0.5;
        rect_tilde_01_syncPhase = 0;
        rect_tilde_02_frequency = 0;
        rect_tilde_02_pulsewidth = 0.5;
        rect_tilde_02_syncPhase = 0;
        ctlin_01_input = 0;
        ctlin_01_controller = 0;
        ctlin_01_channel = -1;
        expr_02_in1 = 0;
        expr_02_in2 = 0.007874015748;
        expr_02_out1 = 0;
        _currentTime = 0;
        audioProcessSampleCount = 0;
        sampleOffsetIntoNextAudioBuffer = 0;
        zeroBuffer = nullptr;
        dummyBuffer = nullptr;
        signals[0] = nullptr;
        signals[1] = nullptr;
        signals[2] = nullptr;
        signals[3] = nullptr;
        signals[4] = nullptr;
        signals[5] = nullptr;
        signals[6] = nullptr;
        signals[7] = nullptr;
        signals[8] = nullptr;
        signals[9] = nullptr;
        signals[10] = nullptr;
        didAllocateSignals = 0;
        vs = 0;
        maxvs = 0;
        sr = 44100;
        invsr = 0.000022675736961451248;
        numberobj_01_currentFormat = 6;
        numberobj_01_lastValue = 0;
        param_01_lastValue = 0;
        noise_tilde_01_setupDone = false;
        cycle_tilde_01_wrap = 0;
        cycle_tilde_01_ph_currentPhase = 0;
        cycle_tilde_01_ph_conv = 0;
        cycle_tilde_01_setupDone = false;
        numbertilde_01_currentInterval = 0;
        numbertilde_01_currentIntervalInSamples = 0;
        numbertilde_01_lastValue = 0;
        numbertilde_01_outValue = 0;
        numbertilde_01_rampInSamples = 0;
        numbertilde_01_currentMode = 0;
        numbertilde_01_smooth_d_prev = 0;
        numbertilde_01_smooth_prev = 0;
        numbertilde_01_smooth_index = 0;
        numbertilde_01_smooth_increment = 0;
        numbertilde_01_setupDone = false;
        saw_tilde_01_t = 0;
        saw_tilde_01_lastSyncPhase = 0;
        saw_tilde_01_lastSyncDiff = 0;
        saw_tilde_01_didSync = false;
        saw_tilde_01_dcblocker_xm1 = 0;
        saw_tilde_01_dcblocker_ym1 = 0;
        saw_tilde_01_setupDone = false;
        numbertilde_02_currentInterval = 0;
        numbertilde_02_currentIntervalInSamples = 0;
        numbertilde_02_lastValue = 0;
        numbertilde_02_outValue = 0;
        numbertilde_02_rampInSamples = 0;
        numbertilde_02_currentMode = 0;
        numbertilde_02_smooth_d_prev = 0;
        numbertilde_02_smooth_prev = 0;
        numbertilde_02_smooth_index = 0;
        numbertilde_02_smooth_increment = 0;
        numbertilde_02_setupDone = false;
        tri_tilde_01_t = 0;
        tri_tilde_01_lastSyncPhase = 0;
        tri_tilde_01_lastSyncDiff = 0;
        tri_tilde_01_didSync = false;
        tri_tilde_01_yn = 0;
        tri_tilde_01_yn1 = 0;
        tri_tilde_01_yn2 = 0;
        tri_tilde_01_yn3 = 0;
        tri_tilde_01_flg = 0;
        tri_tilde_01_app_correction = 0;
        tri_tilde_01_dcblocker_xm1 = 0;
        tri_tilde_01_dcblocker_ym1 = 0;
        tri_tilde_01_setupDone = false;
        ip_01_lastIndex = 0;
        ip_01_lastValue = 0;
        ip_01_resetCount = 0;
        ip_01_sigbuf = nullptr;
        ip_01_setupDone = false;
        rect_tilde_01_xHistory = 0;
        rect_tilde_01_yHistory = 0;
        rect_tilde_01_t = 0;
        rect_tilde_01_lastSyncPhase = 0;
        rect_tilde_01_lastSyncDiff = 0;
        rect_tilde_01_didSync = false;
        rect_tilde_02_xHistory = 0;
        rect_tilde_02_yHistory = 0;
        rect_tilde_02_t = 0;
        rect_tilde_02_lastSyncPhase = 0;
        rect_tilde_02_lastSyncDiff = 0;
        rect_tilde_02_didSync = false;
        ctlin_01_status = 0;
        ctlin_01_byte1 = -1;
        ctlin_01_inchan = 0;
        stackprotect_count = 0;
        _voiceIndex = 0;
        _noteNumber = 0;
        isMuted = 1;
        parameterOffset = 0;
    }
    
    // member variables
    
        number numberobj_01_value;
        number expr_01_in1;
        number expr_01_in2;
        number expr_01_out1;
        number selector_01_onoff;
        number param_01_value;
        number noise_tilde_01_seed;
        number cycle_tilde_01_frequency;
        number cycle_tilde_01_phase_offset;
        number numbertilde_01_input_number;
        number numbertilde_01_ramp;
        number saw_tilde_01_frequency;
        number saw_tilde_01_syncPhase;
        number numbertilde_02_input_number;
        number numbertilde_02_ramp;
        number tri_tilde_01_frequency;
        number tri_tilde_01_pulsewidth;
        number tri_tilde_01_syncPhase;
        number selector_02_onoff;
        number ip_01_value;
        number ip_01_impulse;
        number rect_tilde_01_frequency;
        number rect_tilde_01_pulsewidth;
        number rect_tilde_01_syncPhase;
        number rect_tilde_02_frequency;
        number rect_tilde_02_pulsewidth;
        number rect_tilde_02_syncPhase;
        number ctlin_01_input;
        number ctlin_01_controller;
        number ctlin_01_channel;
        number expr_02_in1;
        number expr_02_in2;
        number expr_02_out1;
        MillisecondTime _currentTime;
        UInt64 audioProcessSampleCount;
        Index sampleOffsetIntoNextAudioBuffer;
        signal zeroBuffer;
        signal dummyBuffer;
        SampleValue * signals[11];
        bool didAllocateSignals;
        Index vs;
        Index maxvs;
        number sr;
        number invsr;
        Int numberobj_01_currentFormat;
        number numberobj_01_lastValue;
        number param_01_lastValue;
        UInt noise_tilde_01_nz_state[4] = { };
        bool noise_tilde_01_setupDone;
        SampleBufferRef cycle_tilde_01_buffer;
        Int cycle_tilde_01_wrap;
        UInt32 cycle_tilde_01_phasei;
        SampleValue cycle_tilde_01_f2i;
        number cycle_tilde_01_ph_currentPhase;
        number cycle_tilde_01_ph_conv;
        bool cycle_tilde_01_setupDone;
        SampleIndex numbertilde_01_currentInterval;
        SampleIndex numbertilde_01_currentIntervalInSamples;
        number numbertilde_01_lastValue;
        number numbertilde_01_outValue;
        number numbertilde_01_rampInSamples;
        Int numbertilde_01_currentMode;
        number numbertilde_01_smooth_d_prev;
        number numbertilde_01_smooth_prev;
        number numbertilde_01_smooth_index;
        number numbertilde_01_smooth_increment;
        bool numbertilde_01_setupDone;
        number saw_tilde_01_t;
        number saw_tilde_01_lastSyncPhase;
        number saw_tilde_01_lastSyncDiff;
        bool saw_tilde_01_didSync;
        number saw_tilde_01_dcblocker_xm1;
        number saw_tilde_01_dcblocker_ym1;
        bool saw_tilde_01_setupDone;
        SampleIndex numbertilde_02_currentInterval;
        SampleIndex numbertilde_02_currentIntervalInSamples;
        number numbertilde_02_lastValue;
        number numbertilde_02_outValue;
        number numbertilde_02_rampInSamples;
        Int numbertilde_02_currentMode;
        number numbertilde_02_smooth_d_prev;
        number numbertilde_02_smooth_prev;
        number numbertilde_02_smooth_index;
        number numbertilde_02_smooth_increment;
        bool numbertilde_02_setupDone;
        number tri_tilde_01_t;
        number tri_tilde_01_lastSyncPhase;
        number tri_tilde_01_lastSyncDiff;
        bool tri_tilde_01_didSync;
        number tri_tilde_01_yn;
        number tri_tilde_01_yn1;
        number tri_tilde_01_yn2;
        number tri_tilde_01_yn3;
        number tri_tilde_01_flg;
        number tri_tilde_01_app_correction;
        number tri_tilde_01_dcblocker_xm1;
        number tri_tilde_01_dcblocker_ym1;
        bool tri_tilde_01_setupDone;
        SampleIndex ip_01_lastIndex;
        number ip_01_lastValue;
        SampleIndex ip_01_resetCount;
        signal ip_01_sigbuf;
        bool ip_01_setupDone;
        number rect_tilde_01_xHistory;
        number rect_tilde_01_yHistory;
        number rect_tilde_01_t;
        number rect_tilde_01_lastSyncPhase;
        number rect_tilde_01_lastSyncDiff;
        bool rect_tilde_01_didSync;
        number rect_tilde_02_xHistory;
        number rect_tilde_02_yHistory;
        number rect_tilde_02_t;
        number rect_tilde_02_lastSyncPhase;
        number rect_tilde_02_lastSyncDiff;
        bool rect_tilde_02_didSync;
        Int ctlin_01_status;
        Int ctlin_01_byte1;
        Int ctlin_01_inchan;
        number stackprotect_count;
        Index _voiceIndex;
        Int _noteNumber;
        Index isMuted;
        ParameterIndex parameterOffset;
        bool _isInitialized = false;
};

		
void advanceTime(EXTERNALENGINE*) {}
void advanceTime(INTERNALENGINE*) {
	_internalEngine.advanceTime(sampstoms(this->vs));
}

void processInternalEvents(MillisecondTime time) {
	_internalEngine.processEventsUntil(time);
}

void updateTime(MillisecondTime time, INTERNALENGINE*, bool inProcess = false) {
	if (time == TimeNow) time = getPatcherTime();
	processInternalEvents(inProcess ? time + sampsToMs(this->vs) : time);
	updateTime(time, (EXTERNALENGINE*)nullptr);
}

rnbomatic* operator->() {
    return this;
}
const rnbomatic* operator->() const {
    return this;
}
rnbomatic* getTopLevelPatcher() {
    return this;
}

void cancelClockEvents()
{
}

template<typename LISTTYPE = list> void listquicksort(LISTTYPE& arr, LISTTYPE& sortindices, Int l, Int h, bool ascending) {
    if (l < h) {
        Int p = (Int)(this->listpartition(arr, sortindices, l, h, ascending));
        this->listquicksort(arr, sortindices, l, p - 1, ascending);
        this->listquicksort(arr, sortindices, p + 1, h, ascending);
    }
}

template<typename LISTTYPE = list> Int listpartition(LISTTYPE& arr, LISTTYPE& sortindices, Int l, Int h, bool ascending) {
    number x = arr[(Index)h];
    Int i = (Int)(l - 1);

    for (Int j = (Int)(l); j <= h - 1; j++) {
        bool asc = (bool)((bool)(ascending) && arr[(Index)j] <= x);
        bool desc = (bool)((bool)(!(bool)(ascending)) && arr[(Index)j] >= x);

        if ((bool)(asc) || (bool)(desc)) {
            i++;
            this->listswapelements(arr, i, j);
            this->listswapelements(sortindices, i, j);
        }
    }

    i++;
    this->listswapelements(arr, i, h);
    this->listswapelements(sortindices, i, h);
    return i;
}

template<typename LISTTYPE = list> void listswapelements(LISTTYPE& arr, Int a, Int b) {
    auto tmp = arr[(Index)a];
    arr[(Index)a] = arr[(Index)b];
    arr[(Index)b] = tmp;
}

number mstosamps(MillisecondTime ms) {
    return ms * this->sr * 0.001;
}

number maximum(number x, number y) {
    return (x < y ? y : x);
}

MillisecondTime sampstoms(number samps) {
    return samps * 1000 / this->sr;
}

MillisecondTime getPatcherTime() const {
    return this->_currentTime;
}

void deallocateSignals() {
    Index i;

    for (i = 0; i < 1; i++) {
        this->signals[i] = freeSignal(this->signals[i]);
    }

    this->globaltransport_tempo = freeSignal(this->globaltransport_tempo);
    this->globaltransport_state = freeSignal(this->globaltransport_state);
    this->zeroBuffer = freeSignal(this->zeroBuffer);
    this->dummyBuffer = freeSignal(this->dummyBuffer);
}

Index getMaxBlockSize() const {
    return this->maxvs;
}

number getSampleRate() const {
    return this->sr;
}

bool hasFixedVectorSize() const {
    return false;
}

void setProbingTarget(MessageTag ) {}

void fillRNBODefaultSinus(DataRef& ref) {
    SampleBuffer buffer(ref);
    number bufsize = buffer->getSize();

    for (Index i = 0; i < bufsize; i++) {
        buffer[i] = rnbo_cos(i * 3.14159265358979323846 * 2. / bufsize);
    }
}

void fillDataRef(DataRefIndex index, DataRef& ref) {
    switch (index) {
    case 0:
        {
        this->fillRNBODefaultSinus(ref);
        break;
        }
    }
}

void allocateDataRefs() {
    this->p_01->allocateDataRefs();

    if (this->RNBODefaultSinus->hasRequestedSize()) {
        if (this->RNBODefaultSinus->wantsFill())
            this->fillRNBODefaultSinus(this->RNBODefaultSinus);

        this->getEngine()->sendDataRefUpdated(0);
    }
}

void initializeObjects() {
    this->p_01->initializeObjects();
}

Index getIsMuted()  {
    return this->isMuted;
}

void setIsMuted(Index v)  {
    this->isMuted = v;
}

void onSampleRateChanged(double ) {}

void extractState(PatcherStateInterface& ) {}

void applyState() {

    this->p_01->setEngineAndPatcher(this->getEngine(), this);
    this->p_01->initialize();
    this->p_01->setParameterOffset(this->getParameterOffset(this->p_01));
}

ParameterIndex getParameterOffset(BaseInterface& subpatcher) const {
    if (addressOf(subpatcher) == addressOf(this->p_01))
        return 0;

    return 0;
}

void processClockEvent(MillisecondTime , ClockId , bool , ParameterValue ) {}

void processOutletAtCurrentTime(EngineLink* , OutletIndex , ParameterValue ) {}

void processOutletEvent(
    EngineLink* sender,
    OutletIndex index,
    ParameterValue value,
    MillisecondTime time
) {
    this->updateTime(time, (ENGINE*)nullptr);
    this->processOutletAtCurrentTime(sender, index, value);
}

void sendOutlet(OutletIndex index, ParameterValue value) {
    this->getEngine()->sendOutlet(this, index, value);
}

void startup() {
    this->updateTime(this->getEngine()->getCurrentTime(), (ENGINE*)nullptr);
    this->p_01->startup();
    this->processParamInitEvents();
}

void p_01_midihandler(int status, int channel, int port, ConstByteArray data, Index length) {
    RNBO_UNUSED(port);
    RNBO_UNUSED(channel);
    RNBO_UNUSED(status);
    this->p_01->processMidiEvent(_currentTime, 0, data, length);
}

void p_01_perform(
    const SampleValue * in1,
    const SampleValue * in2,
    SampleValue * out1,
    SampleValue * out2,
    Index n
) {
    ConstSampleArray<2> ins = {in1, in2};
    SampleArray<2> outs = {out1, out2};
    this->p_01->process(ins, 2, outs, 2, n);
}

void signalforwarder_01_perform(const SampleValue * input, SampleValue * output, Index n) {
    copySignal(output, input, n);
}

void signalforwarder_02_perform(const SampleValue * input, SampleValue * output, Index n) {
    copySignal(output, input, n);
}

void stackprotect_perform(Index n) {
    RNBO_UNUSED(n);
    auto __stackprotect_count = this->stackprotect_count;
    __stackprotect_count = 0;
    this->stackprotect_count = __stackprotect_count;
}

void globaltransport_advance() {}

void globaltransport_dspsetup(bool ) {}

bool stackprotect_check() {
    this->stackprotect_count++;

    if (this->stackprotect_count > 128) {
        console->log("STACK OVERFLOW DETECTED - stopped processing branch !");
        return true;
    }

    return false;
}

Index getPatcherSerial() const {
    return 0;
}

void sendParameter(ParameterIndex index, bool ignoreValue) {
    this->getEngine()->notifyParameterValueChanged(index, (ignoreValue ? 0 : this->getParameterValue(index)), ignoreValue);
}

void scheduleParamInit(ParameterIndex index, Index order) {
    this->paramInitIndices->push(index);
    this->paramInitOrder->push(order);
}

void processParamInitEvents() {
    this->listquicksort(
        this->paramInitOrder,
        this->paramInitIndices,
        0,
        (int)(this->paramInitOrder->length - 1),
        true
    );

    for (Index i = 0; i < this->paramInitOrder->length; i++) {
        this->getEngine()->scheduleParameterBang(this->paramInitIndices[i], 0);
    }
}

void updateTime(MillisecondTime time, EXTERNALENGINE* engine, bool inProcess = false) {
    RNBO_UNUSED(inProcess);
    RNBO_UNUSED(engine);
    this->_currentTime = time;
    auto offset = rnbo_fround(this->msToSamps(time - this->getEngine()->getCurrentTime(), this->sr));

    if (offset >= (SampleIndex)(this->vs))
        offset = (SampleIndex)(this->vs) - 1;

    if (offset < 0)
        offset = 0;

    this->sampleOffsetIntoNextAudioBuffer = (Index)(offset);
}

void assign_defaults()
{
    p_01_target = 0;
    _currentTime = 0;
    audioProcessSampleCount = 0;
    sampleOffsetIntoNextAudioBuffer = 0;
    zeroBuffer = nullptr;
    dummyBuffer = nullptr;
    signals[0] = nullptr;
    didAllocateSignals = 0;
    vs = 0;
    maxvs = 0;
    sr = 44100;
    invsr = 0.000022675736961451248;
    globaltransport_tempo = nullptr;
    globaltransport_state = nullptr;
    stackprotect_count = 0;
    _voiceIndex = 0;
    _noteNumber = 0;
    isMuted = 1;
}

    // data ref strings
    struct DataRefStrings {
    	static constexpr auto& name0 = "RNBODefaultSinus";
    	static constexpr auto& file0 = "";
    	static constexpr auto& tag0 = "buffer~";
    	DataRefStrings* operator->() { return this; }
    	const DataRefStrings* operator->() const { return this; }
    };

    DataRefStrings dataRefStrings;

// member variables

    number p_01_target;
    MillisecondTime _currentTime;
    ENGINE _internalEngine;
    UInt64 audioProcessSampleCount;
    Index sampleOffsetIntoNextAudioBuffer;
    signal zeroBuffer;
    signal dummyBuffer;
    SampleValue * signals[1];
    bool didAllocateSignals;
    Index vs;
    Index maxvs;
    number sr;
    number invsr;
    signal globaltransport_tempo;
    signal globaltransport_state;
    number stackprotect_count;
    DataRef RNBODefaultSinus;
    Index _voiceIndex;
    Int _noteNumber;
    Index isMuted;
    indexlist paramInitIndices;
    indexlist paramInitOrder;
    RNBOSubpatcher_23 p_01;
    bool _isInitialized = false;
};

static PatcherInterface* creaternbomatic()
{
    return new rnbomatic<EXTERNALENGINE>();
}

#ifndef RNBO_NO_PATCHERFACTORY
extern "C" PatcherFactoryFunctionPtr GetPatcherFactoryFunction()
#else
extern "C" PatcherFactoryFunctionPtr rnbomaticFactoryFunction()
#endif
{
    return creaternbomatic;
}

#ifndef RNBO_NO_PATCHERFACTORY
extern "C" void SetLogger(Logger* logger)
#else
void rnbomaticSetLogger(Logger* logger)
#endif
{
    console = logger;
}

} // end RNBO namespace

