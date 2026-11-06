**Govia — Aircraft Track Association Source Notes**

These notes reflect the source context available for the exercise. Optional attributes are not a normalized data dictionary. Some meanings and conventions must be investigated; others may remain unresolved.

**Core Observation Fields**

* **observation\_id:** Unique observation reference. Identical retransmissions reuse it.  
* **tracklet\_id:** Fixed grouping of reports known to belong to one aircraft within that fragment.  
* **observed\_at:** Measurement time, ISO 8601 UTC with Z.  
* **received\_at:** Platform receipt time, ISO 8601 UTC with Z.  
* **latitude / longitude:** Decimal degrees; north/east positive.  
* **altitude\_ft:** Altitude measurement in feet; may be omitted or null.  
* **source:** Namespace identifying the contributing feed.  
* **attributes:** Optional source-specific JSON fields.

All core fields except altitude\_ft and attributes are required. The documented time and position fields are sufficient to attempt the core association task. Preserve original values when deriving or normalizing features.

**Available Source Context**

The following are incomplete integration notes, not verified definitions for every attribute.

**feed-a**

An older movement-report export. Integration notes refer to speed in knots and a directional channel in degrees. Records also contain reference-like values and source diagnostics. There is no complete attribute mapping or validated quality-code interpretation.

**feed-b**

A separate movement feed with shorter channel names. Its numeric conventions were not standardized with feed-a. Fields with matching names must not be assumed to share units or meaning. No complete channel dictionary is available.

**feed-c**

A movement feed from a different adapter. Available notes mention metric speed values but do not specify the unit convention. Some names were inherited from earlier exports. Reference-like values and categorical codes are present, without guarantees about identity or ordering.

**telemetry-x**

An auxiliary stream containing a mixture of sensor-related values and transport diagnostics. Names alone do not establish that a field measures aircraft behavior. No complete channel dictionary is available.

These sources may contain useful, redundant, unreliable, or irrelevant values. **A reference-like value is not an authoritative aircraft identity.** Its persistence and agreement with other evidence should be checked.

**Supporting Records and Joins**

The supporting.jsonl file contains optional records from telemetry-x. Each record has record\_id, observation\_id, source, received\_at, and attributes.

* Join to position reports by **observation\_id** within the same scenario and snapshot.  
* Not every observation has a supporting record.  
* Identical supporting retransmissions reuse **record\_id**; deduplicate before aggregating.  
* Supporting records can arrive later than their linked position reports.  
* Receipt timestamps determine availability, not the time of aircraft movement.

The join keys are defined; inferring undocumented join relationships is not part of the exercise.

**Interpretation Expectations**

Investigate optional attributes using training data and select any resulting features or settings using validation data. **Do not use test labels to decode fields or revise interpretations.**

For any optional field you use, explain:

* What you believe it represents and whether that meaning is documented or inferred.  
* The checks supporting your interpretation, including units where relevant.  
* How you handle missing, inconsistent, or implausible values.  
* Whether it improves association results or provides useful supporting evidence.

Document unresolved assumptions. Do not silently assign physical meaning or units based solely on a name, numeric range, or correlation. You may exclude fields whose meaning or reliability you cannot establish.

**You are not expected to identify every field.** A justified exclusion is acceptable, and using more attributes does not earn additional credit.

